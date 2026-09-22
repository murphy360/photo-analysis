from typing import Protocol

from app.pipeline.types import DescriptionResult

_BASE_PROMPT = (
    "Describe what is happening in this security/trail camera photo in 1-3 concise "
    "sentences. Focus on people, animals, vehicles, and any notable activity (e.g. "
    "a package delivery, someone approaching a door, an animal passing through). If "
    "nothing notable is happening, say so plainly."
)


def build_description_prompt(
    known_people: list[str] | None = None, scene_context: str | None = None
) -> str:
    """CompreFace runs before this (app.pipeline.orchestrator), so by the
    time the vision-LLM is called, identity may already be known — pass
    those names through instead of the model describing "a woman" when we
    already know it's Cathleen Murphy. Without any known names, explicitly
    tell it not to guess, since a wrong guessed name is worse than none.

    scene_context (from a source's sources.yaml entry) is a fixed
    description of that camera's normal, unchanging view — without it, a
    model tends to spend most of its 1-3 sentences re-describing the yard/
    path/trees that look the same in every single photo, instead of the
    thing that actually triggered the capture. Telling it what's normal
    lets it focus on what's different."""
    parts = [_BASE_PROMPT]

    if scene_context:
        parts.append(
            f" This camera's normal, unchanging view: {scene_context} Focus on what's "
            "notable or out of place relative to that — the person, animal, vehicle, "
            "or object that actually triggered this capture — rather than "
            "re-describing the fixed background itself."
        )

    if known_people:
        names = ", ".join(known_people)
        parts.append(
            f" Face recognition has already identified the following people in this "
            f"photo: {names}. Refer to each by that name wherever the description "
            f'would naturally mention them (e.g. "Cathleen Murphy walks a dog" rather '
            f'than "a woman walks a dog") — write it naturally, as if you already knew '
            f"who they were, not as something facial recognition told you."
        )
    else:
        parts.append(
            " Do not speculate about who any person is by name — identity isn't "
            "available for this photo."
        )

    return "".join(parts)


class VisionProvider(Protocol):
    name: str

    async def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        cheap: bool = False,
        known_people: list[str] | None = None,
        scene_context: str | None = None,
    ) -> DescriptionResult: ...
