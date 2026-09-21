import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.models.analysis import AnalysisJob
from app.models.enums import AnalysisTier
from app.pipeline.types import TriageResult

logger = logging.getLogger(__name__)


class SourcePolicy(BaseModel):
    base_tier: AnalysisTier = AnalysisTier.STANDARD
    escalate_tier: AnalysisTier = AnalysisTier.STANDARD
    escalate_on: list[str] = Field(default_factory=lambda: ["person"])
    max_daily_analyses: int = 200
    provider_preference: list[str] = Field(
        default_factory=lambda: ["anthropic", "gemini", "openai", "grok"]
    )

    def escalates(self, triage: TriageResult) -> bool:
        if "*" in self.escalate_on:
            return True
        return bool(triage.labels & set(self.escalate_on))


class PolicyDecision(BaseModel):
    tier: AnalysisTier
    policy: SourcePolicy
    budget_note: str | None = None


_default_policy: SourcePolicy | None = None
_source_policies: dict[str, SourcePolicy] | None = None


def _load_config() -> tuple[SourcePolicy, dict[str, SourcePolicy]]:
    global _default_policy, _source_policies
    if _default_policy is not None and _source_policies is not None:
        return _default_policy, _source_policies

    settings = get_settings()
    path = Path(settings.sources_config_path)
    if not path.exists():
        logger.warning("Sources config %s not found; using built-in defaults only", path)
        _default_policy = SourcePolicy()
        _source_policies = {}
        return _default_policy, _source_policies

    raw = yaml.safe_load(path.read_text()) or {}
    default_raw = raw.get("default", {})
    _default_policy = SourcePolicy.model_validate(default_raw)
    _source_policies = {
        name: SourcePolicy.model_validate({**default_raw, **(cfg or {})})
        for name, cfg in (raw.get("sources") or {}).items()
    }
    return _default_policy, _source_policies


def reload_config() -> None:
    """Drops the cached policy so the next lookup re-reads sources.yaml from
    disk — lets you tune per-camera policy without restarting the service."""
    global _default_policy, _source_policies
    _default_policy = None
    _source_policies = None


def get_policy(source: str) -> SourcePolicy:
    default_policy, policies = _load_config()
    return policies.get(source, default_policy)


async def _analyses_today(session: AsyncSession, source: str) -> int:
    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await session.exec(
        select(func.count(AnalysisJob.id)).where(
            AnalysisJob.source == source, AnalysisJob.created_at >= midnight
        )
    )
    return result.one()


async def decide_tier(
    source: str,
    triage: TriageResult,
    requested_tier: AnalysisTier | None,
    session: AsyncSession,
) -> PolicyDecision:
    """The orchestrator's tasking decision: how much (paid) analysis, if any,
    does this specific image get? Combines the source's configured policy with
    what local triage actually found and today's spend so far for that source."""
    policy = get_policy(source)

    count_today = await _analyses_today(session, source)
    if count_today >= policy.max_daily_analyses:
        return PolicyDecision(
            tier=AnalysisTier.SKIP,
            policy=policy,
            budget_note=(
                f"Daily budget of {policy.max_daily_analyses} analyses for source "
                f"'{source}' reached ({count_today} today); triage-only."
            ),
        )

    if requested_tier is not None:
        # An explicit per-request override (e.g. a human re-running analysis on
        # one photo) wins over the automatic policy, budget permitting.
        return PolicyDecision(tier=requested_tier, policy=policy)

    tier = policy.escalate_tier if policy.escalates(triage) else policy.base_tier
    return PolicyDecision(tier=tier, policy=policy)
