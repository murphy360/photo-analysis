from typing import Protocol

from app.pipeline.types import DescriptionResult

_BASE_PROMPT = (
    "Describe what is happening in this security/trail camera photo in 1-3 concise "
    "sentences. Focus on people, animals, vehicles, and any notable activity (e.g. "
    "a package delivery, someone approaching a door, an animal passing through). If "
    "nothing notable is happening, say so plainly."
)


def build_description_prompt(known_people: list[str] | None = None) -> str:
    """CompreFace runs before this (app.pipeline.orchestrator), so by the
    time the vision-LLM is called, identity may already be known — pass
    those names through instead of the model describing "a woman" when we
    already know it's Cathleen Murphy. Without any known names, explicitly
    tell it not to guess, since a wrong guessed name is worse than none."""
    if known_people:
        names = ", ".join(known_people)
        return (
            f"{_BASE_PROMPT} Face recognition has already identified the following "
            f"people in this photo: {names}. Refer to each by that name wherever the "
            f"description would naturally mention them (e.g. \"Cathleen Murphy walks a "
            f"dog\" rather than \"a woman walks a dog\") — write it naturally, as if you "
            f"already knew who they were, not as something facial recognition told you."
        )
    return (
        f"{_BASE_PROMPT} Do not speculate about who any person is by name — identity "
        "isn't available for this photo."
    )


class VisionProvider(Protocol):
    name: str

    async def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        cheap: bool = False,
        known_people: list[str] | None = None,
    ) -> DescriptionResult: ...
