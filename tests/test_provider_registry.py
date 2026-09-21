from app.providers import registry


class _FakeProvider:
    def __init__(self, name: str) -> None:
        self.name = name


def test_pick_providers_prefers_the_least_used(monkeypatch):
    a, b, c = _FakeProvider("a"), _FakeProvider("b"), _FakeProvider("c")
    monkeypatch.setattr(registry, "get_enabled_providers", lambda: {"a": a, "b": b, "c": c})

    chosen = registry.pick_providers([], 1, {"a": 5, "b": 5, "c": 0})
    assert chosen == [c]


def test_pick_providers_breaks_ties_randomly(monkeypatch):
    a, b = _FakeProvider("a"), _FakeProvider("b")
    monkeypatch.setattr(registry, "get_enabled_providers", lambda: {"a": a, "b": b})

    seen = {registry.pick_providers([], 1, {})[0].name for _ in range(30)}
    # Both tied at 0 uses; over enough trials both must show up, or the
    # "randomize" half of the request isn't actually happening.
    assert seen == {"a", "b"}


def test_pick_providers_still_respects_the_preference_pool(monkeypatch):
    """provider_preference stays a hard filter (e.g. memoire's [anthropic,
    gemini] deliberately excludes openai/grok even if configured) — usage
    only decides ordering *within* whichever pool a provider falls into,
    it doesn't let an unlisted provider jump the preferred ones."""
    a, b, c = _FakeProvider("a"), _FakeProvider("b"), _FakeProvider("c")
    monkeypatch.setattr(registry, "get_enabled_providers", lambda: {"a": a, "b": b, "c": c})

    chosen = registry.pick_providers(["a", "b"], 2, {"a": 10, "b": 10, "c": 0})
    assert {p.name for p in chosen} == {"a", "b"}


def test_pick_providers_tops_up_from_outside_preference_when_short(monkeypatch):
    a, b = _FakeProvider("a"), _FakeProvider("b")
    monkeypatch.setattr(registry, "get_enabled_providers", lambda: {"a": a, "b": b})

    chosen = registry.pick_providers(["a"], 2, {})
    assert {p.name for p in chosen} == {"a", "b"}


def test_pick_providers_raises_when_none_enabled(monkeypatch):
    monkeypatch.setattr(registry, "get_enabled_providers", lambda: {})
    try:
        registry.pick_providers([], 1, {})
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
