from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    photo_service_api_key: str = Field(alias="PHOTO_SERVICE_API_KEY")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="ANTHROPIC_MODEL")
    anthropic_model_cheap: str = Field(
        default="claude-haiku-4-5-20251001", alias="ANTHROPIC_MODEL_CHEAP"
    )

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-flash-latest", alias="GEMINI_MODEL")
    gemini_model_cheap: str = Field(default="gemini-flash-latest", alias="GEMINI_MODEL_CHEAP")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_model_cheap: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_CHEAP")

    grok_api_key: str | None = Field(default=None, alias="GROK_API_KEY")
    grok_base_url: str = Field(default="https://api.x.ai/v1", alias="GROK_BASE_URL")
    grok_model: str = Field(default="grok-2-vision-1212", alias="GROK_MODEL")
    grok_model_cheap: str = Field(default="grok-2-vision-1212", alias="GROK_MODEL_CHEAP")

    compreface_url: str | None = Field(default=None, alias="COMPREFACE_URL")
    compreface_recognition_api_key: str | None = Field(
        default=None, alias="COMPREFACE_RECOGNITION_API_KEY"
    )
    compreface_similarity_threshold: float = Field(
        default=0.85, alias="COMPREFACE_SIMILARITY_THRESHOLD"
    )

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/photo_analysis.db", alias="DATABASE_URL"
    )
    media_dir: str = Field(default="./data/media", alias="MEDIA_DIR")

    sources_config_path: str = Field(
        default="./app/config/sources.yaml", alias="SOURCES_CONFIG_PATH"
    )
    # MegaDetector (via PytorchWildlife) is a camera-trap-specific animal/person/
    # vehicle detector, not a generic COCO model, so it classifies a backyard deer
    # as "animal" reliably instead of guessing "horse" or "cow" the way COCO-YOLO
    # would. Free/local, runs on every image before any paid provider is considered.
    triage_confidence: float = Field(default=0.4, alias="TRIAGE_CONFIDENCE")


@lru_cache
def get_settings() -> Settings:
    return Settings()
