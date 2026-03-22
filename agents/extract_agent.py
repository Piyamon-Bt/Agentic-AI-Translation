import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from models.schemas import AgentResult, AgentStatus


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


def _extract_text_with_ocr(file_path: str) -> str:
    """
    Fallback: render each PDF page as image and run Tesseract OCR.
    Used when the PDF is scanned or has no embedded text layer.
    """
    doc = fitz.open(file_path)
    ocr_results = []

    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better OCR accuracy
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        # OCR with Chinese Simplified + English
        ocr_text = pytesseract.image_to_string(
            image,
            lang="chi_sim+eng",
            config="--psm 6",
        )
        if ocr_text.strip():
            ocr_results.append(ocr_text)

    doc.close()
    return "\n".join(ocr_results)


def _has_sufficient_text(text: str, min_chars: int = 50) -> bool:
    """Check if extracted text meets minimum length threshold."""
    return len(text.strip()) >= min_chars


def run_extract_agent(file_path: str) -> AgentResult:
    """
    Extracts text from a PDF file.
    Attempts native text extraction first, falls back to OCR if needed.
    """
    try:
        extracted_text = _extract_text_with_pymupdf(file_path)
        extraction_method = "native"

        if not _has_sufficient_text(extracted_text):
            extracted_text = _extract_text_with_ocr(file_path)
            extraction_method = "ocr"

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