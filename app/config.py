import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "KrushiVerse-AI — Autonomous Agriculture Platform"
    VERSION: str = "10.0.0"
    DEFAULT_LANGUAGE: str = "mr"  # Marathi by default
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Optional external API settings
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    QDRANT_HOST: str | None = os.getenv("QDRANT_HOST")
    NEO4J_URI: str | None = os.getenv("NEO4J_URI")
    OPENWEATHER_API_KEY: str | None = os.getenv("OPENWEATHER_API_KEY")

settings = Settings()
