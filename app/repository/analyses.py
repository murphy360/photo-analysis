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
