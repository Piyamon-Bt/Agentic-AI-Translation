from langchain.schema import HumanMessage, SystemMessage
from core.gemini_client import get_llm, _call_with_retry
from models.schemas import AgentResult, AgentStatus


SUMMARY_SYSTEM_PROMPT = """You are a document analyst. Given translated English text from a Chinese document,
write a concise, professional summary in English.

Your summary must include:
1. Document purpose and type
2. Key parties involved (if any)
3. Main topics or subject matter
4. Important dates, amounts, or figures (if present)
5. Key conclusions or outcomes

Keep the summary between 100-200 words. Be factual and objective. Output only the summary text."""


def run_summarize_agent(translated_text: str, document_type: str) -> AgentResult:
    """
    Generates a concise English summary of the translated document.
    Uses the translated text (not original Chinese) for better accuracy.
    """
    try:
        llm = get_llm(temperature=0.3)

        # Use first 4000 characters of translation for summarization
        sample_text = translated_text[:4000]
        user_message = (
            f"Document type: {document_type}\n\n"
            f"Translated content:\n{sample_text}\n\n"
            "Write a concise summary of this document."
        )

        summary = _call_with_retry(
            llm,
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ],
        )

        return AgentResult(
            agent_name="summarize_agent",
            status=AgentStatus.DONE,
            output={"summary": summary},
        )

    except Exception as exc:
        return AgentResult(
            agent_name="summarize_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )