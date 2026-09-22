import base64

from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.pipeline.types import DescriptionResult
from app.providers.base import build_description_prompt


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model
        self._model_cheap = settings.anthropic_model_cheap

    async def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        cheap: bool = False,
        known_people: list[str] | None = None,
        scene_context: str | None = None,
        location: str | None = None,
    ) -> DescriptionResult:
        model = self._model_cheap if cheap else self._model
        response = await self._client.messages.create(
            model=model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64.standard_b64encode(image_bytes).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": build_description_prompt(
                                known_people, scene_context, location
                            ),
                        },
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return DescriptionResult(text=text.strip(), provider=self.name)
