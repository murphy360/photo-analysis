import logging

from ultralytics import YOLO

from app.core.config import get_settings
from app.pipeline.types import DetectedObject, TriageResult

logger = logging.getLogger(__name__)

_model: YOLO | None = None

# COCO's 80 classes don't include "deer" (or most wildlife), so a backyard deer
# typically gets reported as one of the quadruped classes below (dog/horse/cow/
# sheep/bear) rather than matched exactly. This maps those specific, sometimes-
# wrong class names down to the three broad buckets app.pipeline.policy actually
# reasons about — bucketed as "animal" either way, which is what the cost
# policy needs to correctly skip the LLM call, even though the specific label
# may be wrong. (A camera-trap-specific detector like MegaDetector would do
# better here; see README for why v1 uses plain COCO-YOLO instead.)
_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}


def _category(label: str) -> str:
    if label == "person":
        return "person"
    if label in _VEHICLE_CLASSES:
        return "vehicle"
    if label in _ANIMAL_CLASSES:
        return "animal"
    return "other"


def load_model() -> YOLO:
    """Loads the triage model once and caches it at module scope. Called from
    the FastAPI lifespan on startup so the (larger) model load never happens
    on the request path."""
    global _model
    if _model is None:
        logger.info("Loading YOLOv8n triage model...")
        _model = YOLO("yolov8n.pt")
    return _model


def run(image_path: str) -> TriageResult:
    """Free, local, always-on first pass: does this image contain a person, an
    animal, a vehicle, or nothing interesting at all? Runs before any paid
    vision provider is considered — app.pipeline.policy uses this to decide
    whether a paid analysis is even warranted (don't burn tokens on a routine
    deer)."""
    settings = get_settings()
    model = load_model()
    results = model.predict(image_path, conf=settings.triage_confidence, verbose=False)

    objects: list[DetectedObject] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls[0])]
            confidence = float(box.conf[0])
            objects.append(
                DetectedObject(label=label, category=_category(label), confidence=confidence)
            )

    return TriageResult(objects=objects)
