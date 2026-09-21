from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisTier(StrEnum):
    """How much (paid) analysis a job gets, decided by policy in
    app.pipeline.policy from the source's config plus what triage found."""

    SKIP = "skip"  # triage only, no face-id, no LLM call
    CHEAP = "cheap"  # one fast/cheap vision-LLM call
    STANDARD = "standard"  # one capable vision-LLM call
    THOROUGH = "thorough"  # multiple providers cross-checked, always runs face-id


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
