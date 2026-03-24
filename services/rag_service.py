import csv
import logging
from pathlib import Path

import chromadb
import fitz
from chromadb.config import Settings as ChromaSettings
from docx import Document as DocxDocument
from google import generativeai as genai

from core.config import settings

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"

GLOSSARY_EXTENSIONS = {".csv"}
REFERENCE_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}

# Chunk settings (ported from reference implementation)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20

COLLECTION_NAME = "chinese_translation_kb"


# ── Gemini embedding (replaces OpenAI embedding) ──────────────────────────────

def _get_gemini_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector using Gemini text-embedding-004.
    Equivalent to get_openai_embedding() in the reference implementation.
    """
    genai.configure(api_key=settings.gemini_api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def _get_gemini_query_embedding(text: str) -> list[float]:
    """
    Generate a query embedding using Gemini text-embedding-004.
    Uses retrieval_query task type for better query-document matching.
    """
    genai.configure(api_key=settings.gemini_api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


# ── ChromaDB client (direct, same pattern as reference) ──────────────────────

def _get_chroma_collection() -> chromadb.Collection:
    """
    Initialize a persistent ChromaDB client and return the collection.
    Equivalent to chroma_client.get_or_create_collection() in the reference.
    """
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Document loading ──────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Ported directly from the reference split_text() function.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def _split_by_paragraph(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text by paragraph boundaries first, then merge small paragraphs
    into chunks that stay under chunk_size. Produces more coherent chunks
    than character-based splitting for document knowledge bases.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph exceeds chunk_size, save current and start new
        if current_chunk and len(current_chunk) + len(paragraph) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk = f"{current_chunk}\n{paragraph}" if current_chunk else paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= 20]


def _extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Same extraction logic as extract_agent.py.
    """
    doc = fitz.open(str(file_path))
    text_parts = []

    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(page_text)

    doc.close()
    full_text = "\n".join(text_parts)

    if len(full_text.strip()) < 50:
        logger.warning("PDF %s has very little text — may be scanned", file_path.name)

    return full_text


def _extract_text_from_docx(file_path: Path) -> str:
    """
    Extract text from a DOCX file using python-docx.
    Preserves paragraph structure for better chunking.
    """
    doc = DocxDocument(str(file_path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _load_documents_from_knowledge_base() -> list[dict]:
    """
    Scan the knowledge_base/ directory and load all supported files.
    Supports: .csv (glossary), .txt/.md (reference), .pdf, .docx

    Equivalent to load_documents_from_directory() in the reference implementation.
    Returns list of dicts with keys: id, text, metadata
    """
    if not KNOWLEDGE_BASE_DIR.exists():
        logger.warning("knowledge_base/ directory not found at %s", KNOWLEDGE_BASE_DIR)
        return []

    raw_documents = []

    for file_path in sorted(KNOWLEDGE_BASE_DIR.iterdir()):
        suffix = file_path.suffix.lower()

        if suffix in GLOSSARY_EXTENSIONS:
            raw_documents.extend(_load_glossary_csv(file_path))

        elif suffix in REFERENCE_EXTENSIONS:
            raw_documents.append({
                "id": file_path.name,
                "text": file_path.read_text(encoding="utf-8"),
                "metadata": {"source": file_path.name, "type": "reference"},
            })

        elif suffix in PDF_EXTENSIONS:
            text = _extract_text_from_pdf(file_path)
            if text.strip():
                raw_documents.append({
                    "id": file_path.name,
                    "text": text,
                    "metadata": {"source": file_path.name, "type": "reference", "format": "pdf"},
                })
            else:
                logger.warning("Skipping empty PDF: %s", file_path.name)

        elif suffix in DOCX_EXTENSIONS:
            text = _extract_text_from_docx(file_path)
            if text.strip():
                raw_documents.append({
                    "id": file_path.name,
                    "text": text,
                    "metadata": {"source": file_path.name, "type": "reference", "format": "docx"},
                })
            else:
                logger.warning("Skipping empty DOCX: %s", file_path.name)

        else:
            logger.debug("Skipping unsupported file: %s", file_path.name)

    logger.info("Loaded %d raw documents from knowledge base", len(raw_documents))
    return raw_documents


def _load_glossary_csv(file_path: Path) -> list[dict]:
    """
    Load glossary CSV and convert each row to a document entry.
    Each term becomes its own document: 'chinese - english'
    """
    documents = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chinese = row.get("chinese", "").strip()
            english = row.get("english", "").strip()
            domain = row.get("domain", "general").strip()

            if not chinese or not english:
                continue

            documents.append({
                "id": f"{file_path.stem}_{chinese}",
                "text": f"{chinese} - {english}",
                "metadata": {
                    "chinese": chinese,
                    "english": english,
                    "domain": domain,
                    "source": file_path.name,
                    "type": "glossary",
                },
            })

    logger.info("Loaded %d glossary terms from %s", len(documents), file_path.name)
    return documents


# ── Seeding (upsert pattern from reference) ───────────────────────────────────

def seed_vector_store() -> None:
    """
    Load documents from knowledge_base/, split into chunks, embed, and upsert
    into ChromaDB. Follows the exact same pipeline as the reference implementation:

    load_documents → split_text → get_embedding → collection.upsert()
    """
    raw_documents = _load_documents_from_knowledge_base()
    if not raw_documents:
        logger.warning("No documents found — skipping seed")
        return

    chunked_documents = []
    for doc in raw_documents:
        doc_type = doc["metadata"]["type"]
        doc_format = doc["metadata"].get("format", "text")

        if doc_type == "glossary":
            # Glossary terms are short — no need to chunk
            chunked_documents.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
            })
        elif doc_format in ("pdf", "docx"):
            # Use paragraph-aware chunking for structured documents
            chunks = _split_by_paragraph(doc["text"])
            logger.info("Split %s (%s) into %d paragraph chunks", doc["id"], doc_format, len(chunks))
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    "id": f"{doc['id']}_chunk{i + 1}",
                    "text": chunk,
                    "metadata": {**doc["metadata"], "chunk_index": i + 1},
                })
        else:
            # Plain text — use character-based chunking (reference pattern)
            chunks = _split_text(doc["text"])
            logger.info("Split %s into %d chunks", doc["id"], len(chunks))
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    "id": f"{doc['id']}_chunk{i + 1}",
                    "text": chunk,
                    "metadata": {**doc["metadata"], "chunk_index": i + 1},
                })

    # Generate embeddings and upsert (same as reference loop)
    collection = _get_chroma_collection()

    for doc in chunked_documents:
        logger.info("Generating embedding for: %s", doc["id"])
        embedding = _get_gemini_embedding(doc["text"])

        # upsert — same as collection.upsert() in reference
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[embedding],
            metadatas=[doc["metadata"]],
        )

    logger.info("Upserted %d chunks into ChromaDB", len(chunked_documents))


def _ensure_seeded() -> chromadb.Collection:
    """Return the collection, seeding it first if it is empty."""
    collection = _get_chroma_collection()
    if collection.count() == 0:
        logger.info("Collection is empty — seeding from knowledge_base/")
        seed_vector_store()
        collection = _get_chroma_collection()
    return collection


# ── Query functions (ported from reference query_documents) ───────────────────

def search_technical_terms(text: str, n_results: int = 10) -> list[dict]:
    """
    Search for technical/glossary terms in the extracted text.

    Strategy:
    1. Exact match — scan text directly for known Chinese terms
    2. Semantic search — query ChromaDB with embedding (same as reference query_documents)
    """
    collection = _ensure_seeded()
    matched_terms = []
    seen_terms: set[str] = set()

    # Step 1: Exact match against all glossary entries
    glossary_result = collection.get(where={"type": "glossary"})
    for metadata in (glossary_result.get("metadatas") or []):
        chinese_term = metadata.get("chinese", "")
        if chinese_term and chinese_term in text and chinese_term not in seen_terms:
            seen_terms.add(chinese_term)
            matched_terms.append({
                "chinese": chinese_term,
                "english": metadata.get("english", ""),
                "domain": metadata.get("domain", "general"),
                "source": metadata.get("source", ""),
            })

    # Step 2: Semantic search — same as collection.query() in reference
    query_embedding = _get_gemini_query_embedding(text[:500])
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"type": "glossary"},
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    for doc_text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
        chinese_term = metadata.get("chinese", "")
        if not chinese_term or chinese_term in seen_terms:
            continue
        # cosine distance < 0.5 means similarity > 0.5
        if distance >= 0.5:
            continue

        seen_terms.add(chinese_term)
        matched_terms.append({
            "chinese": chinese_term,
            "english": metadata.get("english", ""),
            "domain": metadata.get("domain", "general"),
            "source": metadata.get("source", ""),
            "similarity_score": round(1 - float(distance), 4),
        })

    return matched_terms


def search_reference_context(text: str, n_results: int = 3) -> list[str]:
    """
    Retrieve relevant reference document chunks for a given query.
    Equivalent to query_documents() in the reference — returns relevant_chunks.
    Used to inject domain-specific context into the translation prompt.
    """
    collection = _ensure_seeded()

    query_embedding = _get_gemini_query_embedding(text[:500])

    # Same as collection.query() in reference implementation
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"type": "reference"},
    )

    documents = results.get("documents") or [[]]
    distances = results.get("distances") or [[]]

    # Extract relevant chunks (same as reference relevant_chunks)
    context_chunks = []
    for doc_text, distance in zip(documents[0], distances[0]):
        if distance < 0.7:
            context_chunks.append(doc_text)

    return context_chunks


def reset_vector_store() -> None:
    """
    Delete and recreate the collection, clearing all stored embeddings.
    Call this after adding or updating files in knowledge_base/.
    """
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    chroma_client.delete_collection(name=COLLECTION_NAME)
    logger.info("Collection deleted — will reseed on next request")
