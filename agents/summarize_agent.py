from langchain.schema import HumanMessage, SystemMessage
from core.gemini_client import get_llm, _call_with_retry
from models.schemas import AgentResult, AgentStatus


SUMMARY_SYSTEM_PROMPT = """You are a professional document analyst specializing in Thai summaries.
Given the translated Thai text from a Chinese document, write a concise, professional summary in Thai.

Your summary must include:
1. Document purpose and type
2. Key parties involved (if any)
3. Main topics or subject matter
4. Important dates, amounts, or figures (if present)
5. Key conclusions or outcomes

Instructions:
- Write the summary entirely in Thai.
- Keep the summary between 100-200 words.
- Be factual and objective.
- Output the summary as a single, continuous paragraph without any newline characters (\\n).
- Output ONLY the summary text, nothing else."""


def run_summarize_agent(translated_text: str, document_type: str) -> AgentResult:
    """
    Generates a concise Thai summary of the translated document as continuous text.
    """
    try:
        # Use temperature 0.3 for factual and focused summaries
        llm = get_llm(temperature=0.3)

        # Pre-clean translated_text to remove redundant newlines for better context analysis
        clean_context = " ".join(translated_text.split())

        # Use first 4000 characters of translation for summarization
        sample_text = clean_context[:4000]
        user_message = (
            f"Document type: {document_type}\n\n"
            f"Translated content:\n{sample_text}\n\n"
            "สรุปเนื้อหาสำคัญของเอกสารนี้เป็นภาษาไทย โดยให้เขียนเป็นย่อหน้าเดียวต่อเนื่องกัน"
        )

        raw_summary = _call_with_retry(
            llm,
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ],
        )

        # Final cleanup: Ensure no newline characters or extra spaces in the output
        final_summary = " ".join(raw_summary.split())

        return AgentResult(
            agent_name="summarize_agent",
            status=AgentStatus.DONE,
            output={"summary": final_summary},
        )

    except Exception as exc:
        return AgentResult(
            agent_name="summarize_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )