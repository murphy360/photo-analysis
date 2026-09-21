from pydantic import BaseModel


class DetectedObject(BaseModel):
    label: str  # specific class name the detector reported, e.g. "dog", "car", "person"
    category: str  # normalized bucket policy acts on: "person" | "animal" | "vehicle" | "other"
    confidence: float
    box: dict | None = None  # {"x_min", "y_min", "x_max", "y_max"} in source-image pixels


class TriageResult(BaseModel):
    objects: list[DetectedObject]

    @property
    def labels(self) -> set[str]:
        """Normalized categories (person/animal/vehicle/other) — what
        app.pipeline.policy's escalate_on matches against, not the specific
        per-object class names (those stay on DetectedObject.label)."""
        return {o.category for o in self.objects}

    @property
    def is_empty(self) -> bool:
        return not self.objects


class DescriptionResult(BaseModel):
    text: str
    provider: str
