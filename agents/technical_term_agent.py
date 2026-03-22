from services.rag_service import search_technical_terms
from models.schemas import AgentResult, AgentStatus


def run_technical_term_agent(extracted_text: str) -> AgentResult:
    """
    Identifies technical/domain-specific terms in the extracted text.
    Uses RAG (ChromaDB + Gemini embeddings) for both exact and semantic matching.
    """
    try:
        matched_terms = search_technical_terms(extracted_text)

        return AgentResult(
            agent_name="technical_term_agent",
            status=AgentStatus.DONE,
            output={
                "matched_terms": matched_terms,
                "total_found": len(matched_terms),
            },
        )

    except Exception as exc:
        return AgentResult(
            agent_name="technical_term_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )