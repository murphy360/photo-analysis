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

    # CompreFace results. Always attempted now (it's free), but "people" being
    # empty is ambiguous on its own — people_note says which of "not
    # configured" / "ran, found no face" / "errored" actually happened.
    people: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    people_note: str | None = Field(default=None)

    description: str | None = Field(default=None)
    description_providers: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # One entry per vision-LLM provider that actually ran: {"provider", "text"}.
    # Same data description/description_providers summarize, kept separately
    # (not just re-parsed from the merged description string) so a caller —
    # namely /ui's provider-comparison view — can render each one distinctly.
    provider_results: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    # True when this job forced every enabled provider to run for comparison
    # (a /ui-only testing knob — production traffic picks one via tier
    # policy), regardless of what tier_used would normally have dispatched.
    compare_providers: bool = Field(default=False)

    budget_note: str | None = Field(default=None)
    error: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_utcnow, index=True)
    completed_at: datetime | None = Field(default=None)
