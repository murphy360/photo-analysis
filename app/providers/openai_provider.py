import base64

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.pipeline.types import DescriptionResult
from app.providers.base import build_description_prompt


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._model_cheap = settings.openai_model_cheap

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
        data_url = f"data:{mime_type};base64,{base64.standard_b64encode(image_bytes).decode()}"
        response = await self._client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_description_prompt(known_people, scene_context),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
        return DescriptionResult(text=text.strip(), provider=self.name)
