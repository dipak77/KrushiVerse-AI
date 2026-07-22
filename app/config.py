import os
from pydantic import BaseModel


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


class Settings(BaseModel):
    APP_NAME: str = "KrushiVerse-AI — Autonomous Agriculture Platform"
    VERSION: str = "10.2.0"
    DEFAULT_LANGUAGE: str = "mr"  # Marathi by default
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")

    # Advanced RAG / web tools
    ENABLE_WEB_RAG: bool = _env_bool("ENABLE_WEB_RAG", "true")
    ENABLE_TOOL_RAG: bool = _env_bool("ENABLE_TOOL_RAG", "false")
    WEB_CACHE_TTL_SEC: int = int(os.getenv("WEB_CACHE_TTL_SEC", "300"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

    # Mini LLM product integration (Sprint 16 / FP-9)
    # Default ON so all API and UI calls use local model by default
    USE_MINI_LLM: bool = _env_bool("USE_MINI_LLM", "true")
    MINI_DEFAULT_MODE: str = os.getenv("MINI_DEFAULT_MODE", "instruct")
    MINI_MAX_NEW_TOKENS: int = int(os.getenv("MINI_MAX_NEW_TOKENS", "256"))
    MINI_MODEL_VERSION: str = os.getenv("MINI_MODEL_VERSION", "v0.4-agri-qa")
    MINI_TEMPERATURE: float = float(os.getenv("MINI_TEMPERATURE", "0.7"))
    MINI_TOP_P: float = float(os.getenv("MINI_TOP_P", "0.9"))

    # Embeddings + Qdrant
    # backend: auto | hash | minilm | openai
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "auto")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    ENABLE_DENSE_RAG: bool = _env_bool("ENABLE_DENSE_RAG", "true")
    QDRANT_URL: str | None = os.getenv("QDRANT_URL") or (
        f"http://{os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT', '6333')}"
        if os.getenv("QDRANT_HOST")
        else None
    )
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "krushiverse_agri_kb")
    QDRANT_RECREATE: bool = _env_bool("QDRANT_RECREATE", "false")

    # Open data: data.gov.in / Agmarknet-style commodity prices
    DATA_GOV_IN_API_KEY: str | None = os.getenv("DATA_GOV_IN_API_KEY")
    DATA_GOV_IN_BASE_URL: str = os.getenv("DATA_GOV_IN_BASE_URL", "https://api.data.gov.in")
    # Public resource often used for daily Agmarknet commodity prices (override if changed)
    AGMARKNET_RESOURCE_ID: str = os.getenv(
        "AGMARKNET_RESOURCE_ID",
        "9ef84268-d588-465a-a308-a864a43d0070",
    )
    ENABLE_LIVE_AGMARKNET: bool = _env_bool("ENABLE_LIVE_AGMARKNET", "true")
    OPENDATA_CACHE_TTL_SEC: int = int(os.getenv("OPENDATA_CACHE_TTL_SEC", "600"))
    OPENDATA_TIMEOUT_SEC: float = float(os.getenv("OPENDATA_TIMEOUT_SEC", "10"))

    # Optional external API settings
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    XAI_API_KEY: str | None = os.getenv("XAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # When using xAI embeddings-compatible endpoint if available
    XAI_BASE_URL: str = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    QDRANT_HOST: str | None = os.getenv("QDRANT_HOST")
    NEO4J_URI: str | None = os.getenv("NEO4J_URI")
    OPENWEATHER_API_KEY: str | None = os.getenv("OPENWEATHER_API_KEY")


settings = Settings()
