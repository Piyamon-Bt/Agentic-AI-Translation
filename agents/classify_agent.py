import json
from langchain.schema import HumanMessage, SystemMessage
from core.gemini_client import get_llm, _call_with_retry
from models.schemas import AgentResult, AgentStatus, DocumentType

CLASSIFICATION_SYSTEM_PROMPT = """You are a document classification expert specializing in Chinese documents.
Given the extracted text from a Chinese document, classify it into one of these categories:
- contract: Legal agreements, service contracts, employment contracts
- invoice: Bills, payment receipts, purchase orders
- report: Business reports, financial reports, analysis documents
- legal: Court documents, regulations, legal notices
- technical: Technical manuals, specifications, engineering documents
- academic: Research papers, theses, educational materials
- general: General correspondence, articles, memos
- unknown: Cannot be determined

Respond ONLY with a valid JSON object in this exact format (no markdown, no explanation):
{"document_type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief reason in English>"}"""


def run_classify_agent(extracted_text: str) -> AgentResult:
    """
    Classifies the document type using Gemini LLM based only on extracted text.
    Removed technical_terms parameter to match the updated pipeline.
    """
    try:
        # Use low temperature for deterministic classification results
        llm = get_llm(temperature=0.1)

        # Use first 1500 characters which usually contain the most relevant classification context
        sample_text = extracted_text[:1500]
        user_message = f"Classify the following Chinese document:\n\n{sample_text}"

        raw_response = _call_with_retry(
            llm,
            [
                SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ],
        )

        # Parse JSON response from LLM
        # Ensure raw_response is stripped of any potential markdown backticks
        json_str = raw_response.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(json_str)

        doc_type_str = parsed.get("document_type", "unknown")
        
        # Validate and map to DocumentType enum
        try:
            document_type = DocumentType(doc_type_str)
        except ValueError:
            document_type = DocumentType.UNKNOWN

        return AgentResult(
            agent_name="classify_agent",
            status=AgentStatus.DONE,
            output={
                "document_type": document_type.value,
                "confidence": parsed.get("confidence", 0.0),
                "reasoning": parsed.get("reasoning", ""),
            },
        )

    except Exception as exc:
        return AgentResult(
            agent_name="classify_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )