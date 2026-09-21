import json
import mimetypes
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import require_api_key
from app.core.config import get_settings
from app.core.db import get_session
from app.models.analysis import AnalysisJob
from app.models.enums import AnalysisTier, MediaType
from app.pipeline.orchestrator import run_analysis_job
from app.pipeline.video import extract_representative_frame
from app.repository.analyses import list_analyses
from app.schemas.analysis import AnalysisJobResponse

router = APIRouter(prefix="/v1", tags=["analyze"], dependencies=[Depends(require_api_key)])


@router.post("/analyze", response_model=AnalysisJobResponse, status_code=202)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source: str = Form(..., description="Camera/location identifier, e.g. 'front_yard'"),
    tier: AnalysisTier | None = Form(default=None, description="Override the source's policy tier"),
    callback_url: str | None = Form(default=None, description="POSTed the finished job when done"),
    metadata: str | None = Form(default=None, description="JSON object, stored and echoed back"),
    compare_providers: bool = Form(
        default=False,
        description=(
            "Testing only: run every enabled vision-LLM provider and report each one's "
            "answer separately, instead of the tier's usual pick — costs one call per "
            "provider, not meant for production traffic."
        ),
    ),
    session: AsyncSession = Depends(get_session),
) -> AnalysisJobResponse:
    settings = get_settings()
    content_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""
    is_video = content_type.startswith("video/")

    try:
        parsed_metadata = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job = AnalysisJob(
        source=source,
        media_type=MediaType.VIDEO if is_video else MediaType.IMAGE,
        tier_requested=tier,
        callback_url=callback_url,
        metadata_=parsed_metadata,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    media_dir = Path(settings.media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or (".mp4" if is_video else ".jpg")
    original_path = media_dir / f"{job.id}{suffix}"
    original_path.write_bytes(raw_bytes)

    if is_video:
        frame_path = media_dir / f"{job.id}_frame.jpg"
        try:
            extract_representative_frame(original_path, frame_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read video: {exc}") from exc
        job.media_path = str(frame_path)
        job.metadata_ = {**parsed_metadata, "source_video_path": str(original_path)}
    else:
        job.media_path = str(original_path)

    session.add(job)
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(run_analysis_job, job.id, tier, compare_providers)
    return AnalysisJobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=AnalysisJobResponse)
async def get_job(job_id: int, session: AsyncSession = Depends(get_session)) -> AnalysisJobResponse:
    job = await session.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return AnalysisJobResponse.model_validate(job)


@router.get("/analyses", response_model=list[AnalysisJobResponse])
async def get_analyses(
    source: str | None = None,
    since_hours: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[AnalysisJobResponse]:
    jobs = await list_analyses(session, source=source, since_hours=since_hours, limit=limit)
    return [AnalysisJobResponse.model_validate(j) for j in jobs]
