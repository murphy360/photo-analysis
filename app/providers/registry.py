from app.core.config import get_settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import VisionProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.grok_provider import GrokProvider
from app.providers.openai_provider import OpenAIProvider


def get_enabled_providers() -> dict[str, VisionProvider]:
    """Providers are enabled purely by whether their API key is configured, so
    adding a new vendor later is just: write an adapter, add it here, set its
    env var (mirrors trivia_service's provider registry)."""
    settings = get_settings()
    providers: dict[str, VisionProvider] = {}
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider()
    if settings.gemini_api_key:
        providers["gemini"] = GeminiProvider()
    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider()
    if settings.grok_api_key:
        providers["grok"] = GrokProvider()
    return providers


def pick_providers(preference: list[str], count: int) -> list[VisionProvider]:
    """Picks up to `count` providers, in the source's configured preference
    order, from whichever are actually enabled; tops up from any other enabled
    provider if the preferred list doesn't have enough for a thorough/cross-
    check tier."""
    enabled = get_enabled_providers()
    if not enabled:
        raise RuntimeError(
            "No vision-LLM providers configured. Set at least one of "
            "ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, GROK_API_KEY."
        )
    chosen = [enabled[name] for name in preference if name in enabled]
    for provider in enabled.values():
        if len(chosen) >= count:
            break
        if provider not in chosen:
            chosen.append(provider)
    return chosen[:count]
