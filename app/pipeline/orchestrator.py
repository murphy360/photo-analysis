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
from app.pipeline.types import DescriptionResult, TriageResult
from app.providers.registry import get_enabled_providers, pick_providers
from app.schemas.analysis import AnalysisJobResponse

logger = logging.getLogger(__name__)

_TIER_PROVIDER_COUNT = {
    AnalysisTier.CHEAP: 1,
    AnalysisTier.STANDARD: 1,
    AnalysisTier.THOROUGH: 2,
}


async def _run_provider(provider, job_id: int, image_bytes: bytes, mime_type: str, cheap: bool):
    try:
        return await provider.describe(image_bytes, mime_type, cheap=cheap)
    except Exception:
        logger.exception("Vision provider %s failed for job %s", provider.name, job_id)
        return None


async def _describe(job_id: int, image_bytes: bytes, mime_type: str, providers, cheap: bool):
    """Runs the given providers concurrently — the orchestrator's 'divvy out
    tasking' step for the 'what's happening' half of the pipeline. Returns
    one DescriptionResult per provider that succeeded (a failed one is
    logged and silently dropped, not allowed to fail the whole job)."""
    return [
        r
        for r in await asyncio.gather(
            *(_run_provider(p, job_id, image_bytes, mime_type, cheap) for p in providers)
        )
        if r is not None
    ]


def _merge_description(results: list[DescriptionResult]) -> tuple[str | None, list[str]]:
    if not results:
        return None, []
    if len(results) == 1:
        return results[0].text, [results[0].provider]
    merged = "\n".join(f"[{r.provider}] {r.text}" for r in results)
    return merged, [r.provider for r in results]


async def _recognize_people(
    image_bytes: bytes, source_policy: SourcePolicy
) -> tuple[list[dict], str | None]:
    """Returns (people, note). `people` empty is ambiguous on its own — not
    configured, ran and found nothing, or errored all look the same to a
    caller that only checks the list — so `note` says which one happened."""
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


async def _skip_description(triage_result: TriageResult) -> tuple[str, list[str]]:
    if triage_result.is_empty:
        return "Triage only (no LLM call): nothing notable detected.", []
    detail = ", ".join(sorted(f"{o.category} ({o.label})" for o in triage_result.objects))
    return f"Triage only (no LLM call): detected {detail}.", []


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
            # compare_providers is a /ui-only testing knob: run every
            # enabled provider (not just the tier's usual count) so you can
            # see what each one actually says side by side, regardless of
            # what tier policy would normally have picked. Production
            # traffic never sets this — it costs a call per provider.
            if compare_providers:
                description_task = _describe(
                    job_id, image_bytes, mime_type, list(get_enabled_providers().values()), False
                )
                is_skip = False
            elif decision.tier == AnalysisTier.SKIP:
                description_task = _skip_description(triage_result)
                is_skip = True
            else:
                count = _TIER_PROVIDER_COUNT.get(decision.tier, 1)
                providers = pick_providers(decision.policy.provider_preference, count)
                cheap = decision.tier == AnalysisTier.CHEAP
                description_task = _describe(job_id, image_bytes, mime_type, providers, cheap)
                is_skip = False

            (people, people_note), description_result = await asyncio.gather(
                _recognize_people(image_bytes, decision.policy), description_task
            )

            if is_skip:
                job.description, job.description_providers = description_result
                job.provider_results = []
            else:
                job.provider_results = [
                    {"provider": r.provider, "text": r.text} for r in description_result
                ]
                job.description, job.description_providers = _merge_description(
                    description_result
                )

            job.people = people
            job.people_note = people_note

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
