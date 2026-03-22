from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from core.gemini_client import get_embeddings
from core.config import settings


SEED_GLOSSARY = [
    {"chinese": "合同", "english": "Contract", "domain": "legal"},
    {"chinese": "甲方", "english": "Party A", "domain": "legal"},
    {"chinese": "乙方", "english": "Party B", "domain": "legal"},
    {"chinese": "违约金", "english": "Penalty / Liquidated Damages", "domain": "legal"},
    {"chinese": "知识产权", "english": "Intellectual Property", "domain": "legal"},
    {"chinese": "发票", "english": "Invoice", "domain": "finance"},
    {"chinese": "增值税", "english": "Value-Added Tax (VAT)", "domain": "finance"},
    {"chinese": "应收账款", "english": "Accounts Receivable", "domain": "finance"},
    {"chinese": "资产负债表", "english": "Balance Sheet", "domain": "finance"},
    {"chinese": "净利润", "english": "Net Profit", "domain": "finance"},
    {"chinese": "人工智能", "english": "Artificial Intelligence", "domain": "technical"},
    {"chinese": "机器学习", "english": "Machine Learning", "domain": "technical"},
    {"chinese": "深度学习", "english": "Deep Learning", "domain": "technical"},
    {"chinese": "数据库", "english": "Database", "domain": "technical"},
    {"chinese": "应用程序接口", "english": "API (Application Programming Interface)", "domain": "technical"},
    {"chinese": "股东大会", "english": "General Shareholders Meeting", "domain": "corporate"},
    {"chinese": "董事会", "english": "Board of Directors", "domain": "corporate"},
    {"chinese": "年度报告", "english": "Annual Report", "domain": "corporate"},
    {"chinese": "临床试验", "english": "Clinical Trial", "domain": "medical"},
    {"chinese": "病历", "english": "Medical Record", "domain": "medical"},
]


def _build_documents_from_glossary(glossary: list[dict]) -> list[Document]:
    """Convert glossary entries to LangChain Documents for embedding."""
    docs = []
    for entry in glossary:
        content = f"{entry['chinese']} - {entry['english']}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "chinese": entry["chinese"],
                    "english": entry["english"],
                    "domain": entry["domain"],
                },
            )
        )
    return docs


def get_vector_store() -> Chroma:
    """Initialize or load the ChromaDB vector store with glossary data."""
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="technical_glossary",
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_path,
    )

    # Seed the store if it is empty
    if vector_store._collection.count() == 0:
        documents = _build_documents_from_glossary(SEED_GLOSSARY)
        vector_store.add_documents(documents)

    return vector_store


def search_technical_terms(text: str, top_k: int = 10) -> list[dict]:
    """
    Search for technical terms in the extracted text using RAG.
    Returns matched terms with their translations and domain.
    """
    vector_store = get_vector_store()
    matched_terms = []
    seen_terms = set()

    for entry in SEED_GLOSSARY:
        chinese_term = entry["chinese"]
        if chinese_term in text and chinese_term not in seen_terms:
            seen_terms.add(chinese_term)
            matched_terms.append(
                {
                    "chinese": chinese_term,
                    "english": entry["english"],
                    "domain": entry["domain"],
                }
            )

    # Semantic similarity search for terms that were not found by exact match
    results = vector_store.similarity_search_with_score(text[:500], k=top_k)
    for doc, score in results:
        chinese_term = doc.metadata["chinese"]
        if chinese_term not in seen_terms and score < 0.5:
            seen_terms.add(chinese_term)
            matched_terms.append(
                {
                    "chinese": chinese_term,
                    "english": doc.metadata["english"],
                    "domain": doc.metadata["domain"],
                    "similarity_score": round(float(score), 4),
                }
            )

    return matched_terms