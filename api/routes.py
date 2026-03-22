import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.pipeline import run_translation_pipeline
from models.schemas import TranslationResult, ErrorResponse
from core.config import settings


router = APIRouter(prefix="/api/v1", tags=["translation"])


def _save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to disk and return its path."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = uuid.uuid4().hex
    extension = os.path.splitext(upload_file.filename or "file.pdf")[1]
    file_path = os.path.join(settings.upload_dir, f"{file_id}{extension}")

    with open(file_path, "wb") as dest:
        content = upload_file.file.read()
        dest.write(content)

    return file_path


def _cleanup_file(file_path: str) -> None:
    """Remove temporary uploaded file after processing."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


@router.post(
    "/translate",
    response_model=TranslationResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def translate_document(file: UploadFile = File(...)) -> TranslationResult:
    """
    Upload a Chinese PDF document to:
    1. Validate file type
    2. Extract text (OCR fallback for scanned PDFs)
    3. Detect technical/domain terms via RAG
    4. Classify document type
    5. Translate to English
    6. Generate summary
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_path = _save_upload_file(file)

    try:
        result = run_translation_pipeline(file_path, file.filename)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")
    finally:
        _cleanup_file(file_path)