from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.db import engine
from app.models.analysis import AnalysisJob
from app.repository.analyses import provider_usage_counts


async def test_provider_usage_counts_tallies_across_jobs():
    # The test DB is shared across the whole test session (see
    # tests/conftest.py), not reset per test, so other tests' jobs may
    # already be in it — assert on the delta this test itself adds, not
    # on an absolute count.
    async with AsyncSession(engine) as session:
        before = await provider_usage_counts(session)

        session.add(AnalysisJob(source="usage-count-test", description_providers=["anthropic"]))
        session.add(
            AnalysisJob(
                source="usage-count-test", description_providers=["anthropic", "gemini"]
            )
        )
        session.add(AnalysisJob(source="usage-count-test", description_providers=[]))
        await session.commit()

        after = await provider_usage_counts(session)

    assert after.get("anthropic", 0) - before.get("anthropic", 0) == 2
    assert after.get("gemini", 0) - before.get("gemini", 0) == 1
