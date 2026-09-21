import io

from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.pipeline import triage
from app.pipeline.types import DetectedObject, TriageResult
from tests.conftest import FakeProvider

API_KEY = "test-key"


def _fake_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color="blue").save(buf, format="JPEG")
    return buf.getvalue()


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_analyze_requires_api_key():
    async with await _client() as client:
        response = await client.post(
            "/v1/analyze",
            data={"source": "front_yard"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
    assert response.status_code == 401


async def _configure_default_policy(monkeypatch, tmp_path, **overrides) -> None:
    from app.core.config import get_settings

    fields = {
        "base_tier": "skip",
        "escalate_tier": "standard",
        "escalate_on": ["person"],
        "max_daily_analyses": 100,
        **overrides,
    }
    sources_yaml = tmp_path / "sources.yaml"
    lines = ["default:"]
    for key, value in fields.items():
        if isinstance(value, list):
            lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {key}: {value}")
    sources_yaml.write_text("\n".join(lines) + "\n")

    monkeypatch.setenv("SOURCES_CONFIG_PATH", str(sources_yaml))
    get_settings.cache_clear()
    from app.pipeline import policy

    policy.reload_config()


async def test_skip_tier_never_calls_a_vision_provider(monkeypatch, tmp_path):
    """The core cost-control promise: a source whose policy only escalates on
    'person', shown a triage result of just 'animal', must complete with the
    skip tier and never touch a paid vision provider."""
    await _configure_default_policy(monkeypatch, tmp_path)
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="horse", category="animal", confidence=0.9)]
        ),
    )
    fake = FakeProvider("fake")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count: [fake]
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "yard_cam_skip_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        assert create.status_code == 202
        job_id = create.json()["id"]

        # BackgroundTasks run synchronously within the ASGI call above, so the
        # job is already finished by the time we look it up here.
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert job["tier_used"] == "skip"
    assert fake.calls == 0
    assert "animal" in job["description"]


async def test_person_escalates_and_calls_provider(monkeypatch, tmp_path):
    await _configure_default_policy(monkeypatch, tmp_path)
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="person", category="person", confidence=0.95)]
        ),
    )
    fake = FakeProvider("fake", text="A person approaches the front door.")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count: [fake]
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "front_door_escalate_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert job["tier_used"] == "standard"
    assert fake.calls == 1
    assert job["description"] == "A person approaches the front door."
    assert job["description_providers"] == ["fake"]
