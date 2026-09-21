import random

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


def _by_least_used(
    providers: list[VisionProvider], usage_counts: dict[str, int]
) -> list[VisionProvider]:
    # Shuffle first so providers tied on usage (most commonly: everyone at 0,
    # early on) come back in random order rather than always the same one —
    # sort() is stable, so the shuffle is what actually breaks ties.
    shuffled = list(providers)
    random.shuffle(shuffled)
    shuffled.sort(key=lambda p: usage_counts.get(p.name, 0))
    return shuffled


def pick_providers(
    preference: list[str], count: int, usage_counts: dict[str, int] | None = None
) -> list[VisionProvider]:
    """Picks up to `count` providers from whichever are actually enabled,
    preferring ones in the source's configured `preference` list first and
    topping up from any other enabled provider if that list is too short
    for a thorough/cross-check tier. Within each pool, picks whichever
    provider has answered the fewest jobs historically (`usage_counts`,
    from app.repository.analyses.provider_usage_counts), ties broken
    randomly — so load balances across providers over time instead of
    always favoring the first one listed in `preference`."""
    enabled = get_enabled_providers()
    if not enabled:
        raise RuntimeError(
            "No vision-LLM providers configured. Set at least one of "
            "ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, GROK_API_KEY."
        )
    usage_counts = usage_counts or {}

    preferred = _by_least_used(
        [enabled[name] for name in preference if name in enabled], usage_counts
    )
    rest = _by_least_used(
        [p for p in enabled.values() if p not in preferred], usage_counts
    )
    return (preferred + rest)[:count]
