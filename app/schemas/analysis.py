from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import AnalysisTier, JobStatus, MediaType


class AnalyzeMetadata(BaseModel):
    """Free-form fields a caller can attach; stored as-is and returned back.
    Useful for e.g. memoire passing its own object id, or HA passing the
    automation/trigger name."""

    model_config = {"extra": "allow"}


class AnalysisJobResponse(BaseModel):
    id: int
    source: str
    media_type: MediaType
    status: JobStatus
    tier_requested: AnalysisTier | None
    tier_used: AnalysisTier | None
    triage_objects: list[dict]
    people: list[dict]
    people_note: str | None
    description: str | None
    description_providers: list[str]
    provider_results: list[dict]
    compare_providers: bool
    budget_note: str | None
    error: str | None
    # Separate aliases, not one `alias=`: the ORM attribute is `metadata_`
    # (SQLAlchemy reserves `metadata` on declarative models), but FastAPI
    # serializes response models by alias by default, so a single shared
    # alias would leak "metadata_" into the JSON response instead of the
    # clean "metadata" key callers actually want.
    metadata: dict = Field(validation_alias="metadata_", serialization_alias="metadata")
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True, "populate_by_name": True}
