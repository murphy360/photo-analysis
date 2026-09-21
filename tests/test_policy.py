from app.models.enums import AnalysisTier
from app.pipeline import policy
from app.pipeline.types import DetectedObject, TriageResult


def test_escalates_on_matching_label():
    p = policy.SourcePolicy(escalate_on=["person"])
    assert p.escalates(
        TriageResult(objects=[DetectedObject(label="person", category="person", confidence=0.9)])
    )
    assert not p.escalates(
        TriageResult(objects=[DetectedObject(label="dog", category="animal", confidence=0.9)])
    )


def test_escalates_on_wildcard_always():
    p = policy.SourcePolicy(escalate_on=["*"])
    assert p.escalates(TriageResult(objects=[]))


def test_get_policy_falls_back_to_default(monkeypatch, tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "default:\n"
        "  base_tier: skip\n"
        "  escalate_tier: cheap\n"
        "  escalate_on: [person]\n"
        "  max_daily_analyses: 5\n"
        "sources:\n"
        "  front_door:\n"
        "    base_tier: cheap\n"
    )
    from app.core.config import get_settings

    monkeypatch.setenv("SOURCES_CONFIG_PATH", str(sources_yaml))
    get_settings.cache_clear()
    policy.reload_config()

    default = policy.get_policy("some_unconfigured_camera")
    assert default.base_tier == AnalysisTier.SKIP
    assert default.max_daily_analyses == 5

    front_door = policy.get_policy("front_door")
    assert front_door.base_tier == AnalysisTier.CHEAP
    # Unset fields inherit from `default`, so escalate_tier still comes through.
    assert front_door.escalate_tier == AnalysisTier.CHEAP

    get_settings.cache_clear()
    policy.reload_config()
