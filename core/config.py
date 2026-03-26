from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    # gemini_model: str = "gemini-2.5-flash"
    gemini_model: str = "gemini-2.5-flash-lite"
    chroma_db_path: str = "./chroma_db"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 20
    allowed_mime_types: list[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp"
    ]

    class Config:
        env_file = ".env"


settings = Settings()