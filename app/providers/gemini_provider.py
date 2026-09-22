from google import genai
from google.genai import types

from app.core.config import get_settings
from app.pipeline.types import DescriptionResult
from app.providers.base import build_description_prompt


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._model_cheap = settings.gemini_model_cheap

    async def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        cheap: bool = False,
        known_people: list[str] | None = None,
        scene_context: str | None = None,
    ) -> DescriptionResult:
        model = self._model_cheap if cheap else self._model
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                build_description_prompt(known_people, scene_context),
            ],
        )
        return DescriptionResult(text=(response.text or "").strip(), provider=self.name)
