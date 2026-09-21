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
from app.pipeline.types import DescriptionResult
from app.providers.registry import pick_providers
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


async def _describe(
    job_id: int, image_bytes: bytes, mime_type: str, tier: AnalysisTier, preference: list[str]
) -> tuple[str | None, list[str]]:
    """Dispatches the description task to one provider (cheap/standard) or
    several run concurrently and cross-checked (thorough) — the orchestrator's
    'divvy out tasking' step for the 'what's happening' half of the pipeline."""
    count = _TIER_PROVIDER_COUNT.get(tier, 1)
    providers = pick_providers(preference, count)
    cheap = tier == AnalysisTier.CHEAP

    results: list[DescriptionResult] = [
        r
        for r in await asyncio.gather(
            *(_run_provider(p, job_id, image_bytes, mime_type, cheap) for p in providers)
        )
        if r is not None
    ]
    if not results:
        return None, []
    if len(results) == 1:
        return results[0].text, [results[0].provider]

    merged = "\n".join(f"[{r.provider}] {r.text}" for r in results)
    return merged, [r.provider for r in results]


async def _recognize_people(image_bytes: bytes) -> list[dict]:
    client = CompreFaceClient()
    if not client.configured:
        return []
    try:
        return await client.recognize(image_bytes)
    except Exception:
        logger.exception("CompreFace recognition failed")
        return []


async def _noop_people() -> list[dict]:
    return []


async def run_analysis_job(job_id: int, requested_tier: AnalysisTier | None) -> None:
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

            if decision.tier == AnalysisTier.SKIP:
                if triage_result.is_empty:
                    job.description = "Triage only (no LLM call): nothing notable detected."
                else:
                    detail = ", ".join(
                        sorted(f"{o.category} ({o.label})" for o in triage_result.objects)
                    )
                    job.description = f"Triage only (no LLM call): detected {detail}."
                job.description_providers = []
            else:
                run_face_id = (
                    "person" in triage_result.labels or decision.tier == AnalysisTier.THOROUGH
                )
                people, (description, providers_used) = await asyncio.gather(
                    _recognize_people(image_bytes) if run_face_id else _noop_people(),
                    _describe(
                        job_id, image_bytes, mime_type, decision.tier, decision.policy.provider_preference
                    ),
                )
                job.people = people
                job.description = description
                job.description_providers = providers_used

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
