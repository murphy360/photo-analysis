import re
from io import BytesIO

import httpx
from PIL import Image

from app.core.config import get_settings


def _crop_face(image_bytes: bytes, box: dict | None) -> bytes:
    """Crops to just the detected face (with a little padding) before
    enrolling it, rather than submitting the whole source photo — CompreFace
    runs its own detector on whatever we send it, so an uncropped multi-person
    photo could enroll the wrong face under the new subject."""
    if not box:
        return image_bytes
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    x_min, y_min, x_max, y_max = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
    pad_x, pad_y = (x_max - x_min) * 0.15, (y_max - y_min) * 0.15
    crop_box = (
        max(0, int(x_min - pad_x)),
        max(0, int(y_min - pad_y)),
        min(width, int(x_max + pad_x)),
        min(height, int(y_max + pad_y)),
    )
    buf = BytesIO()
    image.crop(crop_box).save(buf, format="JPEG")
    return buf.getvalue()


class CompreFaceClient:
    """Thin client for a self-hosted CompreFace recognition service. Identity
    (the 'who') is intentionally left entirely to CompreFace's own subject
    gallery rather than reimplemented here."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = (settings.compreface_url or "").rstrip("/")
        self._api_key = settings.compreface_recognition_api_key
        self._threshold = settings.compreface_similarity_threshold

    @property
    def configured(self) -> bool:
        return bool(self._base_url and self._api_key)

    async def recognize(self, image_bytes: bytes) -> list[dict]:
        """Returns one entry per detected face: {"subject", "similarity", "box"}.
        A face whose best match falls below the configured similarity threshold
        is reported as subject "unknown" rather than dropped, so the caller still
        knows a person was there even when CompreFace can't name them."""
        if not self.configured:
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/api/v1/recognition/recognize",
                headers={"x-api-key": self._api_key},
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                params={"limit": 1, "det_prob_threshold": 0.8},
            )
            response.raise_for_status()
            data = response.json()

        people = []
        for face in data.get("result", []):
            subjects = face.get("subjects", [])
            best = subjects[0] if subjects else None
            if best and best.get("similarity", 0) >= self._threshold:
                subject, similarity = best["subject"], best["similarity"]
            else:
                subject, similarity = "unknown", (best or {}).get("similarity", 0.0)
            people.append({"subject": subject, "similarity": similarity, "box": face.get("box")})
        return people

    async def enroll_face(self, image_bytes: bytes, box: dict | None, label: str = "Unknown") -> str:
        """Registers an unrecognized face as a brand-new subject in CompreFace's
        collection, so a later sighting of the same person matches this
        placeholder (instead of being enrolled again) even before anyone gives
        them a real name. Rename the subject in CompreFace's own UI once you
        know who it is — this service never renames or merges subjects itself.
        `label` names the placeholder series (e.g. "Amazon Driver" for a
        front-door source you're tracking recurring delivery drivers on) —
        subjects come out as "<label> 1", "<label> 2", etc. Returns the
        generated subject name."""
        crop = _crop_face(image_bytes, box)

        async with httpx.AsyncClient(timeout=30.0) as client:
            subject = await self._next_subject_name(client, label)
            response = await client.post(
                f"{self._base_url}/api/v1/recognition/faces",
                headers={"x-api-key": self._api_key},
                params={"subject": subject},
                files={"file": ("face.jpg", crop, "image/jpeg")},
            )
            response.raise_for_status()
        return subject

    async def _next_subject_name(self, client: httpx.AsyncClient, label: str) -> str:
        """CompreFace's subject list is the source of truth for numbering
        (rather than a counter this service keeps itself), so it stays correct
        across restarts and even if subjects were added by hand in between."""
        response = await client.get(
            f"{self._base_url}/api/v1/recognition/subjects",
            headers={"x-api-key": self._api_key},
        )
        response.raise_for_status()
        existing = response.json().get("subjects", [])

        pattern = re.compile(rf"^{re.escape(label)} (\d+)$")
        highest = 0
        for name in existing:
            match = pattern.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"{label} {highest + 1}"
