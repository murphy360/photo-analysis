from datetime import datetime, timezone

from sqlalchemy import Column
from sqlmodel import JSON, Field, SQLModel

from app.models.enums import AnalysisTier, JobStatus, MediaType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisJob(SQLModel, table=True):
    __tablename__ = "analysis_jobs"

    id: int | None = Field(default=None, primary_key=True)

    # Camera/location identifier the caller reports, e.g. "front_yard", "driveway",
    # or a project name like "memoire" for non-camera uploads. Drives both the
    # per-source cost policy (app.pipeline.policy) and later history queries like
    # "who was at the front door today".
    source: str = Field(index=True)
    media_type: MediaType = Field(default=MediaType.IMAGE)
    media_path: str | None = Field(default=None)

    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    tier_requested: AnalysisTier | None = Field(default=None)
    tier_used: AnalysisTier | None = Field(default=None)
    callback_url: str | None = Field(default=None)
    metadata_: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))

    # Free local object-detection pass (app.pipeline.triage). Always runs.
    triage_objects: list[dict] = Field(default_factory=list, sa_column=Column(JSON))

    # CompreFace results, only populated when policy decided to run face-id.
    people: list[dict] = Field(default_factory=list, sa_column=Column(JSON))

    description: str | None = Field(default=None)
    description_providers: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    budget_note: str | None = Field(default=None)
    error: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_utcnow, index=True)
    completed_at: datetime | None = Field(default=None)
