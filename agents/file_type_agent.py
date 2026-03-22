import magic
from pathlib import Path
from models.schemas import AgentResult, AgentStatus


ALLOWED_MIME_TYPES = {"application/pdf"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def run_file_type_agent(file_path: str, file_name: str) -> AgentResult:
    """
    Validates the uploaded file:
    - Checks MIME type using libmagic (not just extension)
    - Checks file size limit
    """
    path = Path(file_path)

    if not path.exists():
        return AgentResult(
            agent_name="file_type_agent",
            status=AgentStatus.ERROR,
            output={"error": f"File not found: {file_path}"},
        )

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        return AgentResult(
            agent_name="file_type_agent",
            status=AgentStatus.ERROR,
            output={"error": f"File size {file_size} exceeds limit of {MAX_FILE_SIZE_BYTES} bytes"},
        )

    mime_type = magic.from_file(file_path, mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        return AgentResult(
            agent_name="file_type_agent",
            status=AgentStatus.ERROR,
            output={"error": f"Unsupported file type: {mime_type}. Only PDF is allowed."},
        )

    return AgentResult(
        agent_name="file_type_agent",
        status=AgentStatus.DONE,
        output={
            "file_name": file_name,
            "mime_type": mime_type,
            "file_size_bytes": file_size,
        },
    )