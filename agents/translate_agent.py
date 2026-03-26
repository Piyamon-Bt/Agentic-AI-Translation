from langchain.schema import HumanMessage, SystemMessage
from core.gemini_client import get_llm, _call_with_retry
from models.schemas import AgentResult, AgentStatus, DocumentType


def _build_translation_prompt(document_type: str) -> str:
    """Build a context-aware Chinese-to-Thai translation system prompt."""
    
    doc_type_guidance = {
        DocumentType.CONTRACT: "This is a legal contract. Preserve all clause numbering, party names, and legal phrasing precisely.",
        DocumentType.INVOICE: "This is a financial invoice. Keep all numbers, dates, and monetary amounts exactly as-is.",
        DocumentType.REPORT: "This is a business report. Maintain formal tone and preserve all statistical figures.",
        DocumentType.LEGAL: "This is a legal document. Use formal legal Thai and do not paraphrase legal terms.",
        DocumentType.TECHNICAL: "This is a technical document. Preserve all technical terminology, model numbers, and specifications.",
        DocumentType.ACADEMIC: "This is an academic document. Maintain academic tone and preserve all citations and terminology.",
    }.get(document_type, "Translate accurately and maintain the original structure.")

    return f"""You are a professional Chinese-to-Thai translator specializing in formal documents.

Instructions:
1. Translate the Chinese text to Thai accurately and completely.
2. {doc_type_guidance}
3. Preserve the original document structure and paragraphs.
4. Output the translation as continuous text without unnecessary newline characters.
5. Do NOT add explanations, annotations, or translator notes.

Output only the translated Thai text, nothing else."""


def run_translate_agent(
    extracted_text: str,
    document_type: str,
) -> AgentResult:
    """
    Translates Chinese document text to Thai using Gemini and cleans up newline characters.
    """
    try:
        llm = get_llm(temperature=0.2)

        system_prompt = _build_translation_prompt(document_type)

        # 1. Clean extracted_text before processing to remove redundant newlines
        clean_input_text = " ".join(extracted_text.split())

        chunk_size = 3000
        text_chunks = [
            clean_input_text[i: i + chunk_size]
            for i in range(0, len(clean_input_text), chunk_size)
        ]

        translated_parts = []
        for chunk in text_chunks:
            translated_chunk = _call_with_retry(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Translate this Chinese text to Thai:\n\n{chunk}"),
                ],
            )
            translated_parts.append(translated_chunk)

        # 2. Join translated parts and perform final cleanup of any remaining newlines
        combined_translation = " ".join(translated_parts)
        final_translation = " ".join(combined_translation.split())

        return AgentResult(
            agent_name="translate_agent",
            status=AgentStatus.DONE,
            output={
                "translated_text": final_translation,
                "chunk_count": len(text_chunks),
            },
        )

    except Exception as exc:
        return AgentResult(
            agent_name="translate_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )