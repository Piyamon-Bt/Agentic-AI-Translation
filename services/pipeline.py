from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.file_type_agent import run_file_type_agent
from agents.extract_agent import run_extract_agent
from agents.technical_term_agent import run_technical_term_agent
from agents.classify_agent import run_classify_agent
from agents.translate_agent import run_translate_agent
from agents.summarize_agent import run_summarize_agent
from models.schemas import AgentResult, AgentStatus, DocumentType, TranslationResult


class PipelineState(TypedDict):
    file_path: str
    file_name: str
    agent_results: list[AgentResult]
    extracted_text: str
    technical_terms: list[dict]
    document_type: str
    document_type_confidence: float
    translated_text: str
    summary: str
    error: str | None


def node_file_type_check(state: PipelineState) -> PipelineState:
    result = run_file_type_agent(state["file_path"], state["file_name"])
    state["agent_results"].append(result)
    if result.status == AgentStatus.ERROR:
        state["error"] = result.output.get("error")
    return state


def node_extract_text(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    result = run_extract_agent(state["file_path"])
    state["agent_results"].append(result)
    if result.status == AgentStatus.ERROR:
        state["error"] = result.output.get("error")
    else:
        state["extracted_text"] = result.output["extracted_text"]
    return state


def node_check_technical_terms(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    result = run_technical_term_agent(state["extracted_text"])
    state["agent_results"].append(result)
    if result.status == AgentStatus.DONE:
        state["technical_terms"] = result.output.get("matched_terms", [])
    return state


def node_classify_document(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    result = run_classify_agent(state["extracted_text"], state["technical_terms"])
    state["agent_results"].append(result)
    if result.status == AgentStatus.ERROR:
        state["document_type"] = DocumentType.UNKNOWN.value
        state["document_type_confidence"] = 0.0
    else:
        state["document_type"] = result.output.get("document_type", DocumentType.UNKNOWN.value)
        state["document_type_confidence"] = result.output.get("confidence", 0.0)
    return state


def node_translate(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    result = run_translate_agent(
        state["extracted_text"],
        state["document_type"],
        state["technical_terms"],
    )
    state["agent_results"].append(result)
    if result.status == AgentStatus.ERROR:
        state["error"] = result.output.get("error")
    else:
        state["translated_text"] = result.output["translated_text"]
    return state


def node_summarize(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    result = run_summarize_agent(state["translated_text"], state["document_type"])
    state["agent_results"].append(result)
    if result.status == AgentStatus.DONE:
        state["summary"] = result.output.get("summary", "")
    return state


def should_stop_on_error(state: PipelineState) -> str:
    """Conditional edge: stop pipeline if a critical error occurred."""
    if state.get("error"):
        return "end"
    return "continue"


def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("file_type_check", node_file_type_check)
    graph.add_node("extract_text", node_extract_text)
    graph.add_node("check_technical_terms", node_check_technical_terms)
    graph.add_node("classify_document", node_classify_document)
    graph.add_node("translate", node_translate)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("file_type_check")

    graph.add_conditional_edges(
        "file_type_check",
        should_stop_on_error,
        {"end": END, "continue": "extract_text"},
    )
    graph.add_conditional_edges(
        "extract_text",
        should_stop_on_error,
        {"end": END, "continue": "check_technical_terms"},
    )
    graph.add_edge("check_technical_terms", "classify_document")
    graph.add_edge("classify_document", "translate")
    graph.add_conditional_edges(
        "translate",
        should_stop_on_error,
        {"end": END, "continue": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


def run_translation_pipeline(file_path: str, file_name: str) -> TranslationResult:
    """
    Entry point to run the full translation pipeline.
    Returns a TranslationResult with outputs from all agents.
    """
    initial_state: PipelineState = {
        "file_path": file_path,
        "file_name": file_name,
        "agent_results": [],
        "extracted_text": "",
        "technical_terms": [],
        "document_type": DocumentType.UNKNOWN.value,
        "document_type_confidence": 0.0,
        "translated_text": "",
        "summary": "",
        "error": None,
    }

    pipeline = build_pipeline()
    final_state = pipeline.invoke(initial_state)

    if final_state.get("error"):
        raise ValueError(final_state["error"])

    return TranslationResult(
        file_name=file_name,
        document_type=DocumentType(final_state["document_type"]),
        document_type_confidence=final_state["document_type_confidence"],
        extracted_text=final_state["extracted_text"],
        technical_terms=final_state["technical_terms"],
        summary=final_state["summary"],
        translated_text=final_state["translated_text"],
        agent_results=final_state["agent_results"],
    )