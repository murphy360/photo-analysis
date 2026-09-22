import asyncio
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.db import engine
from app.integrations.compreface import CompreFaceClient
from app.models.analysis import AnalysisJob
from app.models.enums import AnalysisTier, JobStatus
from app.pipeline import policy, triage
from app.pipeline.policy import SourcePolicy
from app.pipeline.types import TriageResult
from app.providers.registry import get_enabled_providers, pick_providers
from app.repository.analyses import provider_usage_counts
from app.schemas.analysis import AnalysisJobResponse

logger = logging.getLogger(__name__)

_TIER_PROVIDER_COUNT = {
    AnalysisTier.CHEAP: 1,
    AnalysisTier.STANDARD: 1,
    AnalysisTier.THOROUGH: 2,
}


async def _run_provider(
    provider,
    job_id: int,
    image_bytes: bytes,
    mime_type: str,
    cheap: bool,
    known_people: list[str] | None,
    scene_context: str | None,
) -> dict:
    """Returns {"provider", "text", "error"} either way — a failure is
    logged and never allowed to fail the whole job, but (unlike dropping it
    silently) it stays visible as its own entry instead of just vanishing,
    so e.g. a misconfigured model name shows up as "grok: 404 ..." in
    provider_results rather than an unexplained empty result."""
    try:
        result = await provider.describe(
            image_bytes,
            mime_type,
            cheap=cheap,
            known_people=known_people,
            scene_context=scene_context,
        )
        return {"provider": provider.name, "text": result.text, "error": None}
    except Exception as exc:
        logger.exception("Vision provider %s failed for job %s", provider.name, job_id)
        return {"provider": provider.name, "text": None, "error": str(exc)}


async def _describe(
    job_id: int,
    image_bytes: bytes,
    mime_type: str,
    providers,
    cheap: bool,
    known_people: list[str] | None,
    scene_context: str | None,
) -> list[dict]:
    """Runs the given providers concurrently — the orchestrator's 'divvy out
    tasking' step for the 'what's happening' half of the pipeline. Returns
    one entry per provider *attempted*, success or failure alike."""
    return list(
        await asyncio.gather(
            *(
                _run_provider(
                    p, job_id, image_bytes, mime_type, cheap, known_people, scene_context
                )
                for p in providers
            )
        )
    )


def _merge_description(results: list[dict]) -> tuple[str | None, list[str]]:
    successes = [r for r in results if r["error"] is None]
    if not successes:
        return None, []
    if len(successes) == 1:
        return successes[0]["text"], [successes[0]["provider"]]
    merged = "\n".join(f"[{r['provider']}] {r['text']}" for r in successes)
    return merged, [r["provider"] for r in successes]


async def _recognize_people(
    image_bytes: bytes, source_policy: SourcePolicy
) -> tuple[list[dict], str | None]:
    """Returns (people, note). `people` empty is ambiguous on its own — not
    configured, ran and found nothing, or errored all look the same to a
    caller that only checks the list — so `note` says which one happened.
    Runs *before* the vision-LLM description step (not concurrently with
    it) specifically so a recognized name can be passed into that prompt."""
    client = CompreFaceClient()
    if not client.configured:
        return [], "CompreFace not configured."
    try:
        people = await client.recognize(image_bytes)
    except Exception as exc:
        logger.exception("CompreFace recognition failed")
        return [], f"CompreFace error: {exc}"

    note = None if people else "CompreFace ran; no face detected in this image."

    if not source_policy.auto_enroll_unknown_faces:
        return people, note

    enrolled: list[dict] = []
    for person in people:
        if person["subject"] != "unknown":
            enrolled.append(person)
            continue
        try:
            subject = await client.enroll_face(
                image_bytes, person.get("box"), source_policy.unknown_face_label
            )
            enrolled.append({**person, "subject": subject, "newly_enrolled": True})
        except Exception:
            logger.exception("Failed to auto-enroll unknown face in CompreFace")
            enrolled.append(person)
    return enrolled, note


def _known_names(people: list[dict]) -> list[str]:
    """Every named subject worth mentioning by name in a description —
    CompreFace's literal "unknown" placeholder excluded, but an
    auto-enrolled placeholder like "Amazon Driver 1" is still more useful
    than "a man in a uniform" and stays in."""
    return [p["subject"] for p in people if p["subject"] != "unknown"]


async def _skip_description(triage_result: TriageResult, known_people: list[str]) -> tuple[str, list[str]]:
    if triage_result.is_empty:
        text = "Triage only (no LLM call): nothing notable detected."
    else:
        detail = ", ".join(sorted(f"{o.category} ({o.label})" for o in triage_result.objects))
        text = f"Triage only (no LLM call): detected {detail}."
    if known_people:
        text += f" Recognized: {', '.join(known_people)}."
    return text, []


async def run_analysis_job(
    job_id: int, requested_tier: AnalysisTier | None, compare_providers: bool = False
) -> None:
    callback_url: str | None = None
    callback_payload: dict | None = None

    async with AsyncSession(engine) as session:
        job = await session.get(AnalysisJob, job_id)
        if job is None:
            logger.error("AnalysisJob %s not found", job_id)
            return

        # Captured before the commit below, which (with expire_on_commit's
        # default of True) expires every attribute on `job` — a bare
        # attribute read afterward would try an implicit lazy-load and blow
        # up with MissingGreenlet since that can't happen outside an await.
        # These two never change after job creation, so a snapshot is safe.
        media_path = job.media_path
        source = job.source

        job.status = JobStatus.RUNNING
        job.compare_providers = compare_providers
        session.add(job)
        await session.commit()

        try:
            image_bytes = Path(media_path).read_bytes()
            mime_type = mimetypes.guess_type(media_path)[0] or "image/jpeg"

            triage_result = triage.run(media_path)
            job.triage_objects = [o.model_dump() for o in triage_result.objects]

            decision = await policy.decide_tier(source, triage_result, requested_tier, session)
            job.tier_used = decision.tier
            job.budget_note = decision.budget_note

            # CompreFace and local triage are both free (self-hosted, no
            # per-call cost), so face-id always runs regardless of tier —
            # only the paid vision-LLM description step is tier-gated. This
            # also means a person triage's object detector missed (partial
            # occlusion, small/distant, misclassified) still gets a chance
            # at being identified.
            #
            # Run *before* the description step, not concurrently with it:
            # a name CompreFace recognizes gets passed into the vision-LLM
            # prompt so the description says "Cathleen Murphy" instead of
            # "a woman" — that dependency is exactly why these two can't
            # run in parallel the way they used to.
            people, people_note = await _recognize_people(image_bytes, decision.policy)
            known_people = _known_names(people)
            job.people = people
            job.people_note = people_note

            # compare_providers is a /ui-only testing knob: run every
            # enabled provider (not just the tier's usual count) so you can
            # see what each one actually says side by side, regardless of
            # what tier policy would normally have picked. Production
            # traffic never sets this — it costs a call per provider.
            if compare_providers:
                results = await _describe(
                    job_id,
                    image_bytes,
                    mime_type,
                    list(get_enabled_providers().values()),
                    False,
                    known_people,
                    decision.policy.scene_context,
                )
                job.provider_results = results
                job.description, job.description_providers = _merge_description(results)
            elif decision.tier == AnalysisTier.SKIP:
                job.description, job.description_providers = await _skip_description(
                    triage_result, known_people
                )
                job.provider_results = []
            else:
                count = _TIER_PROVIDER_COUNT.get(decision.tier, 1)
                usage_counts = await provider_usage_counts(session)
                providers = pick_providers(
                    decision.policy.provider_preference, count, usage_counts
                )
                cheap = decision.tier == AnalysisTier.CHEAP
                results = await _describe(
                    job_id,
                    image_bytes,
                    mime_type,
                    providers,
                    cheap,
                    known_people,
                    decision.policy.scene_context,
                )
                job.provider_results = results
                job.description, job.description_providers = _merge_description(results)

            job.status = JobStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001 - surfaced to the client via job.error
            logger.exception("AnalysisJob %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)
            session.add(job)
            await session.commit()
            await session.refresh(job)
            callback_url = job.callback_url
            if callback_url:
                callback_payload = AnalysisJobResponse.model_validate(job).model_dump(mode="json")

    if callback_url and callback_payload:
        await _fire_callback(job_id, callback_url, callback_payload)


async def _fire_callback(job_id: int, callback_url: str, payload: dict) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(callback_url, json=payload)
    except Exception:
        logger.exception("Callback to %s failed for job %s", callback_url, job_id)
