from typing import Protocol

from app.pipeline.types import DescriptionResult

DESCRIPTION_PROMPT = (
    "Describe what is happening in this security/trail camera photo in 1-3 concise "
    "sentences. Focus on people, animals, vehicles, and any notable activity (e.g. "
    "a package delivery, someone approaching a door, an animal passing through). If "
    "nothing notable is happening, say so plainly. Do not speculate about who any "
    "person is by name — a separate face-recognition step handles identity."
)


class VisionProvider(Protocol):
    name: str

    async def describe(
        self, image_bytes: bytes, mime_type: str, *, cheap: bool = False
    ) -> DescriptionResult: ...
