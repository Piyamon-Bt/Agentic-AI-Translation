import magic
from pathlib import Path
from models.schemas import AgentResult, AgentStatus


# Supported MIME types grouped by category
PDF_MIME_TYPES = {"application/pdf"}
IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
}
ALLOWED_MIME_TYPES = PDF_MIME_TYPES | IMAGE_MIME_TYPES

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def _resolve_file_category(mime_type: str) -> str:
    """Return 'pdf' or 'image' based on MIME type."""
    if mime_type in PDF_MIME_TYPES:
        return "pdf"
    if mime_type in IMAGE_MIME_TYPES:
        return "image"
    return "unknown"


def run_file_type_agent(file_path: str, file_name: str) -> AgentResult:
    """
    Validates the uploaded file:
    - Checks MIME type using libmagic (not just extension)
    - Checks file size limit
    - Supports PDF and common image formats (JPEG, PNG, TIFF, BMP, WebP)
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
            output={
                "error": (
                    f"Unsupported file type: {mime_type}. "
                    "Supported formats: PDF, JPEG, PNG, TIFF, BMP, WebP."
                )
            },
        )

    return AgentResult(
        agent_name="file_type_agent",
        status=AgentStatus.DONE,
        output={
            "file_name": file_name,
            "mime_type": mime_type,
            "file_category": _resolve_file_category(mime_type),
            "file_size_bytes": file_size,
        },
    )