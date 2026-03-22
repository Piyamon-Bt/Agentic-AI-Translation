from langchain.schema import HumanMessage, SystemMessage
from core.gemini_client import get_llm, _call_with_retry
from models.schemas import AgentResult, AgentStatus, DocumentType


def _build_translation_prompt(document_type: str, technical_terms: list[dict]) -> str:
    """Build a context-aware translation system prompt with glossary injection."""
    glossary_section = ""
    if technical_terms:
        glossary_lines = [
            f"  - {t['chinese']} = {t['english']}"
            for t in technical_terms[:20]
        ]
        glossary_section = "\n\nMandatory glossary (use these exact translations):\n" + "\n".join(glossary_lines)

    doc_type_guidance = {
        DocumentType.CONTRACT: "This is a legal contract. Preserve all clause numbering, party names, and legal phrasing precisely.",
        DocumentType.INVOICE: "This is a financial invoice. Keep all numbers, dates, and monetary amounts exactly as-is.",
        DocumentType.REPORT: "This is a business report. Maintain formal tone and preserve all statistical figures.",
        DocumentType.LEGAL: "This is a legal document. Use formal legal English and do not paraphrase legal terms.",
        DocumentType.TECHNICAL: "This is a technical document. Preserve all technical terminology, model numbers, and specifications.",
        DocumentType.ACADEMIC: "This is an academic document. Maintain academic tone and preserve all citations and terminology.",
    }.get(document_type, "Translate accurately and maintain the original structure.")

    return f"""You are a professional Chinese-to-English translator specializing in formal documents.

Instructions:
1. Translate the Chinese text to English accurately and completely.
2. {doc_type_guidance}
3. Preserve the original document structure, paragraphs, and formatting.
4. Do NOT add explanations, annotations, or translator notes.
5. If a section is already in English, keep it unchanged.{glossary_section}

Output only the translated English text, nothing else."""


def run_translate_agent(
    extracted_text: str,
    document_type: str,
    technical_terms: list[dict],
) -> AgentResult:
    """
    Translates Chinese document text to English using Gemini.
    Injects technical term glossary for domain-accurate translation.
    Processes text in chunks if it exceeds the recommended input length.
    """
    try:
        llm = get_llm(temperature=0.2)

        system_prompt = _build_translation_prompt(document_type, technical_terms)

        # Process in chunks of 3000 characters to avoid token limits
        chunk_size = 3000
        text_chunks = [
            extracted_text[i: i + chunk_size]
            for i in range(0, len(extracted_text), chunk_size)
        ]

        translated_parts = []
        for chunk in text_chunks:
            translated_chunk = _call_with_retry(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Translate this Chinese text:\n\n{chunk}"),
                ],
            )
            translated_parts.append(translated_chunk)

        full_translation = "\n\n".join(translated_parts)

        return AgentResult(
            agent_name="translate_agent",
            status=AgentStatus.DONE,
            output={
                "translated_text": full_translation,
                "chunk_count": len(text_chunks),
            },
        )

    except Exception as exc:
        return AgentResult(
            agent_name="translate_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )