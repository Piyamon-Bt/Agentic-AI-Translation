from langchain.schema import HumanMessage, SystemMessage
from core.gemini_client import get_llm, _call_with_retry
from models.schemas import AgentResult, AgentStatus, DocumentType
from utils.text_processor import clean_continuous_text

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


def run_translate_agent(extracted_text: str, document_type: str) -> AgentResult:
    try:
        llm = get_llm(temperature=0.2)
        system_prompt = _build_translation_prompt(document_type)

        # 1. Clean input ก่อนส่งเข้า LLM
        clean_input = clean_continuous_text(extracted_text) #

        text_chunks = [
            clean_input[i: i + 3000]
            for i in range(0, len(clean_input), 3000)
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

        # 2. Final Clean เพื่อกำจัด \n ที่ LLM อาจจะเผลอใส่มา
        full_translation = " ".join(translated_parts)
        final_translation = clean_continuous_text(full_translation) #

        return AgentResult(
            agent_name="translate_agent",
            status=AgentStatus.DONE,
            output={"translated_text": final_translation},
        )
    except Exception as exc:
        return AgentResult(agent_name="translate_agent", 
                           status=AgentStatus.ERROR, 
                           output={"error": str(exc)})