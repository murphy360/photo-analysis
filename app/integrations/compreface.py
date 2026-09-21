import httpx

from app.core.config import get_settings


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
