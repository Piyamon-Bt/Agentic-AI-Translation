import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.pipeline import run_translation_pipeline
from services.rag_service import reset_vector_store, _get_chroma_collection
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


@router.get("/knowledge-base/status")
def knowledge_base_status() -> dict:
    """
    Return the number of documents currently stored in the ChromaDB collection.
    """
    try:
        collection = _get_chroma_collection()
        total = collection.count()

        glossary_result = collection.get(where={"type": "glossary"})
        glossary_count = len(glossary_result.get("ids") or [])

        reference_result = collection.get(where={"type": "reference"})
        reference_count = len(reference_result.get("ids") or [])

        return {
            "total_documents": total,
            "glossary_terms": glossary_count,
            "reference_chunks": reference_count,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/knowledge-base/reset")
def knowledge_base_reset() -> dict:
    """
    Delete and recreate the ChromaDB collection.
    Call this after adding or updating files in knowledge_base/.
    """
    try:
        reset_vector_store()
        return {"message": "Vector store reset. It will reseed on the next translation request."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
