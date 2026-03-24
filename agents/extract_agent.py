import base64
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from pathlib import Path

import google.generativeai as genai

from core.config import settings
from models.schemas import AgentResult, AgentStatus

DIRECT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

EXTENSION_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

IMAGE_EXTRACT_PROMPT = (
    "This image contains Chinese text. "
    "Extract ALL Chinese characters exactly as they appear. "
    "Output only the Chinese text, preserving line breaks. "
    "Do NOT translate, romanize, or add any explanation."
)


def _extract_text_from_image_with_gemini(file_path: str) -> str:
    """
    Extract Chinese text from an image using Gemini Vision.
    Far more accurate than Tesseract for mixed Chinese/pinyin layouts,
    handwriting, low-contrast text, and decorative fonts.
    """
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)

    suffix = Path(file_path).suffix.lower()
    mime_type = EXTENSION_TO_MIME.get(suffix, "image/jpeg")

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    image_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode("utf-8"),
    }

    response = model.generate_content([IMAGE_EXTRACT_PROMPT, image_part])
    return response.text.strip()


def _extract_text_with_pymupdf(file_path: str) -> str:
    """Extract embedded text from PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text_parts = []

    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(page_text)

    doc.close()
    return "\n".join(text_parts)


def _extract_text_from_pdf_with_ocr(file_path: str) -> str:
    """
    Fallback: render each PDF page as image and run Tesseract OCR.
    Used when the PDF is scanned or has no embedded text layer.
    """
    doc = fitz.open(file_path)
    ocr_results = []

    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        ocr_text = pytesseract.image_to_string(
            image,
            lang="chi_sim+eng",
            config="--psm 6",
        )
        if ocr_text.strip():
            ocr_results.append(ocr_text)

    doc.close()
    return "\n".join(ocr_results)


def _is_image_file(file_path: str) -> bool:
    """Check if the file is a direct image (not PDF)."""
    return Path(file_path).suffix.lower() in DIRECT_IMAGE_EXTENSIONS


def _has_sufficient_text(text: str, min_chars: int = 50) -> bool:
    """Check if extracted text meets minimum length threshold."""
    return len(text.strip()) >= min_chars


def run_extract_agent(file_path: str) -> AgentResult:
    """
    Extracts Chinese text from a PDF or image file.

    For images (JPEG, PNG, TIFF, BMP, WebP):
      Uses Gemini Vision — handles decorative fonts, mixed pinyin/Chinese,
      low contrast, and complex layouts far better than Tesseract.

    For PDF:
      1. Attempt native text extraction with PyMuPDF
      2. Fall back to Tesseract OCR if text layer is absent or too short
    """
    try:
        if _is_image_file(file_path):
            extracted_text = _extract_text_from_image_with_gemini(file_path)
            extraction_method = "gemini_vision"
        else:
            extracted_text = _extract_text_with_pymupdf(file_path)
            extraction_method = "native"

            if not _has_sufficient_text(extracted_text):
                extracted_text = _extract_text_from_pdf_with_ocr(file_path)
                extraction_method = "pdf_ocr"

        if not _has_sufficient_text(extracted_text):
            return AgentResult(
                agent_name="extract_agent",
                status=AgentStatus.ERROR,
                output={"error": "Could not extract sufficient text from the document."},
            )

        return AgentResult(
            agent_name="extract_agent",
            status=AgentStatus.DONE,
            output={
                "extracted_text": extracted_text,
                "extraction_method": extraction_method,
                "char_count": len(extracted_text),
            },
        )

    except Exception as exc:
        return AgentResult(
            agent_name="extract_agent",
            status=AgentStatus.ERROR,
            output={"error": str(exc)},
        )