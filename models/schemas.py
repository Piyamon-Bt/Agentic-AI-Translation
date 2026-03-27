from pydantic import BaseModel
from enum import Enum


class DocumentType(str, Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    LEGAL = "legal"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    GENERAL = "general"
    UNKNOWN = "unknown"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class AgentResult(BaseModel):
    agent_name: str
    status: AgentStatus
    output: dict


class TranslationResult(BaseModel):
    file_name: str
    document_type: DocumentType
    document_type_confidence: float
    extracted_text: str
    pinyin: str
    # technical_terms: list[dict]
    summary: str
    translated_text: str
    agent_results: list[AgentResult]


class ErrorResponse(BaseModel):
    detail: str
    agent: str | None = None