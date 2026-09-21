from datetime import datetime, timedelta, timezone

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.analysis import AnalysisJob


async def list_analyses(
    session: AsyncSession,
    *,
    source: str | None = None,
    since_hours: int | None = None,
    limit: int = 50,
) -> list[AnalysisJob]:
    query = select(AnalysisJob).order_by(AnalysisJob.created_at.desc()).limit(limit)
    if source is not None:
        query = query.where(AnalysisJob.source == source)
    if since_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        query = query.where(AnalysisJob.created_at >= cutoff)
    result = await session.exec(query)
    return list(result.all())


async def provider_usage_counts(session: AsyncSession) -> dict[str, int]:
    """How many jobs each vision-LLM provider has actually answered, across
    all history — app.providers.registry.pick_providers uses this to favor
    whichever provider has been used least, so usage balances out over time
    instead of always favoring the first one listed in a source's
    provider_preference. Counted from description_providers (one entry per
    provider that actually answered a job), not provider_results, so a
    provider that failed mid-call doesn't count as "used"."""
    result = await session.exec(select(AnalysisJob.description_providers))
    counts: dict[str, int] = {}
    for providers in result.all():
        for name in providers or []:
            counts[name] = counts.get(name, 0) + 1
    return counts
