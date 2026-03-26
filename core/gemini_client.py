import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import BaseMessage
from core.config import settings

logger = logging.getLogger(__name__)

# Free tier allows 5 RPM, so we wait at least 12s between calls to stay safe
# MIN_SECONDS_BETWEEN_CALLS = 12
# MAX_RETRIES = 4
# BASE_BACKOFF_SECONDS = 30

MIN_SECONDS_BETWEEN_CALLS = 3
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 10


def _call_with_retry(llm: ChatGoogleGenerativeAI, messages: list[BaseMessage]) -> str:
    """
    Invoke a Gemini LLM call with exponential backoff on 429 rate limit errors.
    Adds a minimum delay between calls to respect free tier limits.
    """
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS)
            response = llm.invoke(messages)
            return response.content.strip()

        except Exception as exc:
            error_message = str(exc)
            is_rate_limit = "429" in error_message or "quota" in error_message.lower()

            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait_seconds = BASE_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                    attempt + 1,
                    MAX_RETRIES,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            raise

    raise RuntimeError("Max retries exceeded due to rate limiting.")


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create a Gemini embeddings instance."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.gemini_api_key,
    )