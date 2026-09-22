import io

from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.pipeline import triage
from app.pipeline.types import DetectedObject, TriageResult
from tests.conftest import FakeProvider


class FakeCompreFaceClient:
    """Stands in for a real CompreFace deployment: one unrecognized face and
    one already-known one, so auto-enroll tests can assert on both paths
    without any network mocking. enroll_calls is a class attribute (not
    instance) since app.pipeline.orchestrator constructs a fresh client per
    job — tests read it right after the job completes and don't reuse it."""

    enroll_calls: list[tuple[str, dict | None]] = []

    def __init__(self) -> None:
        self.configured = True

    async def recognize(self, image_bytes: bytes) -> list[dict]:
        return [
            {"subject": "unknown", "similarity": 0.0, "box": {"x_min": 0, "y_min": 0, "x_max": 8, "y_max": 8}},
            {"subject": "Corey", "similarity": 0.95, "box": {"x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30}},
        ]

    async def enroll_face(self, image_bytes: bytes, box: dict | None, label: str) -> str:
        FakeCompreFaceClient.enroll_calls.append((label, box))
        return f"{label} {len(FakeCompreFaceClient.enroll_calls)}"

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
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake]
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
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake]
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


async def test_unknown_face_auto_enrolled_with_sequential_label(monkeypatch, tmp_path):
    """A source with auto_enroll_unknown_faces on: an unrecognized face gets
    enrolled under "<unknown_face_label> <n>" and flagged newly_enrolled,
    while an already-known face passes through untouched — no enroll call
    for it."""
    await _configure_default_policy(
        monkeypatch,
        tmp_path,
        base_tier="standard",
        escalate_tier="standard",
        auto_enroll_unknown_faces=True,
        unknown_face_label="Amazon Driver",
    )
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="person", category="person", confidence=0.9)]
        ),
    )
    fake = FakeProvider("fake", text="A driver drops off a package.")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake]
    )
    FakeCompreFaceClient.enroll_calls = []
    monkeypatch.setattr("app.pipeline.orchestrator.CompreFaceClient", FakeCompreFaceClient)

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "front_door_enroll_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert FakeCompreFaceClient.enroll_calls == [("Amazon Driver", {"x_min": 0, "y_min": 0, "x_max": 8, "y_max": 8})]

    people_by_subject = {p["subject"]: p for p in job["people"]}
    assert people_by_subject["Amazon Driver 1"]["newly_enrolled"] is True
    assert "Corey" in people_by_subject
    assert people_by_subject["Corey"].get("newly_enrolled") is not True


async def test_compreface_runs_even_on_skip_tier(monkeypatch, tmp_path):
    """CompreFace and local triage are both free, so face-id must run on
    every image regardless of tier — including skip, where triage found
    only an animal and no paid LLM call happens at all. This catches a
    person the object detector missed (occlusion, distance, misclassified)."""
    await _configure_default_policy(monkeypatch, tmp_path)  # base_tier: skip
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="dog", category="animal", confidence=0.9)]
        ),
    )
    monkeypatch.setattr("app.pipeline.orchestrator.CompreFaceClient", FakeCompreFaceClient)

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "yard_cam_skip_still_recognizes_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert job["tier_used"] == "skip"
    assert job["description_providers"] == []  # no paid LLM call
    assert {p["subject"] for p in job["people"]} == {"unknown", "Corey"}
    # Even the free auto-generated skip-tier text names a recognized person
    # rather than staying purely category-level, since it costs nothing more.
    assert "Recognized: Corey" in job["description"]


async def test_recognized_name_passed_into_the_description_prompt(monkeypatch, tmp_path):
    """The actual point of running CompreFace before the LLM call rather than
    concurrently with it: a recognized name reaches the vision provider's
    known_people argument, so the prompt can say "Cathleen Murphy" instead
    of "a woman" -- and the literal "unknown" placeholder never leaks in."""
    await _configure_default_policy(
        monkeypatch, tmp_path, base_tier="standard", escalate_tier="standard"
    )
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="person", category="person", confidence=0.9)]
        ),
    )
    fake = FakeProvider("fake", text="Corey walks up to the door.")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake]
    )
    monkeypatch.setattr("app.pipeline.orchestrator.CompreFaceClient", FakeCompreFaceClient)

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "known_name_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert job["description"] == "Corey walks up to the door."
    assert fake.known_people_seen == [["Corey"]]  # "unknown" excluded


async def test_people_note_explains_an_unconfigured_compreface(monkeypatch, tmp_path):
    """No CompreFace URL/key set in this test env, so people must come back
    empty with a note saying why -- not silently indistinguishable from
    "ran and found nothing" or "errored"."""
    await _configure_default_policy(monkeypatch, tmp_path)
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(objects=[]),
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "people_note_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["people"] == []
    assert job["people_note"] == "CompreFace not configured."


async def test_compare_providers_runs_every_enabled_provider(monkeypatch, tmp_path):
    """compare_providers=true is a /ui-only testing override: even on a
    source that would normally skip the LLM call entirely, it must run every
    enabled provider and report each one's answer separately in
    provider_results, regardless of the tier policy decided."""
    await _configure_default_policy(monkeypatch, tmp_path)  # base_tier: skip
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="dog", category="animal", confidence=0.9)]
        ),
    )
    fake_a = FakeProvider("fake-a", text="Provider A's take.")
    fake_b = FakeProvider("fake-b", text="Provider B's take.")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.get_enabled_providers",
        lambda: {"fake-a": fake_a, "fake-b": fake_b},
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "compare_providers_test", "compare_providers": "true"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert job["tier_used"] == "skip"  # policy decision is unchanged...
    assert job["compare_providers"] is True
    assert fake_a.calls == 1
    assert fake_b.calls == 1
    results_by_provider = {r["provider"]: r["text"] for r in job["provider_results"]}
    assert results_by_provider == {
        "fake-a": "Provider A's take.",
        "fake-b": "Provider B's take.",
    }


async def test_failed_provider_shows_up_in_provider_results_instead_of_vanishing(
    monkeypatch, tmp_path
):
    """The actual production bug this test exists for: Grok's configured
    model returned 404 for every call, and the job just came back with no
    description and no indication why — the failing provider silently
    disappeared instead of being reported. A failed provider must appear in
    provider_results with its error, and a provider that did succeed must
    still show up correctly alongside it."""
    await _configure_default_policy(monkeypatch, tmp_path)
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(objects=[]),
    )
    fake_ok = FakeProvider("fake-ok", text="A calm, empty yard.")
    fake_broken = FakeProvider("fake-broken", error="404 Not Found: model does not exist")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.get_enabled_providers",
        lambda: {"fake-ok": fake_ok, "fake-broken": fake_broken},
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "provider_failure_test", "compare_providers": "true"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    results_by_provider = {r["provider"]: r for r in job["provider_results"]}
    assert results_by_provider["fake-broken"]["error"] == "404 Not Found: model does not exist"
    assert results_by_provider["fake-broken"]["text"] is None
    assert results_by_provider["fake-ok"]["error"] is None
    assert results_by_provider["fake-ok"]["text"] == "A calm, empty yard."
    # The merged description only draws from providers that actually succeeded.
    assert job["description"] == "A calm, empty yard."
    assert job["description_providers"] == ["fake-ok"]


async def test_all_providers_failing_leaves_description_none_not_a_crash(monkeypatch, tmp_path):
    await _configure_default_policy(
        monkeypatch, tmp_path, base_tier="standard", escalate_tier="standard"
    )
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(objects=[]),
    )
    fake_broken = FakeProvider("fake-broken", error="404 Not Found")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake_broken]
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "provider_total_failure_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"  # a provider failing doesn't fail the whole job
    assert job["description"] is None
    assert job["description_providers"] == []
    assert job["provider_results"] == [
        {"provider": "fake-broken", "text": None, "error": "404 Not Found"}
    ]


async def test_scene_context_reaches_the_provider(monkeypatch, tmp_path):
    """A source's fixed scene_context (sources.yaml) must reach the vision
    provider so the prompt can steer it toward what's different rather than
    re-describing the same yard/path/trees on every single photo."""
    await _configure_default_policy(
        monkeypatch,
        tmp_path,
        base_tier="standard",
        escalate_tier="standard",
        scene_context="a grassy front yard with a gravel path and a metal yard sculpture",
    )
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="person", category="person", confidence=0.9)]
        ),
    )
    fake = FakeProvider("fake")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake]
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "scene_context_test"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert fake.scene_context_seen == [
        "a grassy front yard with a gravel path and a metal yard sculpture"
    ]


async def test_source_name_passed_to_provider_as_location(monkeypatch, tmp_path):
    """The camera/source name itself (e.g. "Front Yard") must reach the
    vision provider too, even when no scene_context is configured for it,
    so the model knows which camera it's looking at."""
    await _configure_default_policy(
        monkeypatch, tmp_path, base_tier="standard", escalate_tier="standard"
    )
    monkeypatch.setattr(
        triage,
        "run",
        lambda path: TriageResult(
            objects=[DetectedObject(label="person", category="person", confidence=0.9)]
        ),
    )
    fake = FakeProvider("fake")
    monkeypatch.setattr(
        "app.pipeline.orchestrator.pick_providers", lambda preference, count, usage_counts=None: [fake]
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/v1/analyze",
            headers=headers,
            data={"source": "Front Yard"},
            files={"file": ("photo.jpg", _fake_jpeg(), "image/jpeg")},
        )
        job_id = create.json()["id"]
        response = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        job = response.json()

    assert job["status"] == "completed"
    assert fake.location_seen == ["Front Yard"]
