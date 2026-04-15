"""
Jobs API - upload video, create dubbing jobs, track progress.
"""
import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.models.job import Job, JobStatus
from app.config import settings

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobResponse(BaseModel):
    id: str
    status: str
    progress: float
    current_stage: str
    original_filename: str
    source_language: str
    target_language: str
    speakers_detected: int
    duration_seconds: Optional[float]
    output_video_path: Optional[str]
    output_srt_path: Optional[str]
    output_original_srt_path: Optional[str]
    output_vtt_path: Optional[str]
    output_original_vtt_path: Optional[str]
    error_message: Optional[str]
    processing_time_seconds: Optional[float]
    created_at: str
    transcription_data: Optional[dict]

    class Config:
        from_attributes = True


ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv"}
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB


@router.post("/", response_model=JobResponse)
async def create_job(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    voice_cloning_enabled: bool = Form(True),
    export_srt: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    if source_language not in settings.supported_languages:
        raise HTTPException(400, f"Unsupported source language: {source_language}")
    if target_language not in settings.supported_languages:
        raise HTTPException(400, f"Unsupported target language: {target_language}")
    if source_language == target_language:
        raise HTTPException(400, "Source and target languages must differ")

    job_id = str(uuid.uuid4())
    upload_path = settings.uploads_dir / f"{job_id}{suffix}"

    async with aiofiles.open(upload_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    job = Job(
        id=job_id,
        original_filename=file.filename,
        input_path=str(upload_path),
        source_language=source_language,
        target_language=target_language,
        voice_cloning_enabled=int(voice_cloning_enabled),
        export_srt=int(export_srt),
        status=JobStatus.PENDING,
        progress=0.0,
        current_stage="pending",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from app.workers.pipeline import run_dubbing_pipeline
    task = run_dubbing_pipeline.apply_async(args=[job_id], queue="dubbing")

    job.celery_task_id = task.id
    await db.commit()
    await db.refresh(job)

    return _job_to_response(job)


@router.get("/", response_model=list[JobResponse])
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job).order_by(desc(Job.created_at)).limit(limit).offset(offset)
    )
    jobs = result.scalars().all()
    return [_job_to_response(j) for j in jobs]


@router.delete("/")
async def clear_all_jobs(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    result = await db.execute(select(Job))
    jobs = result.scalars().all()
    for job in jobs:
        if job.celery_task_id:
            try:
                from app.core.celery_app import celery_app
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            except Exception:
                pass
    await db.execute(delete(Job))
    await db.commit()
    return {"message": f"Cleared {len(jobs)} jobs"}


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_to_response(job)


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(400, "Only failed or cancelled jobs can be retried")

    if not job.input_path or not os.path.exists(job.input_path):
        raise HTTPException(400, "Original video file no longer exists — please re-upload")

    job.status = JobStatus.PENDING
    job.progress = 0.0
    job.current_stage = "pending"
    job.error_message = None
    job.output_video_path = None
    job.output_srt_path = None
    job.output_original_srt_path = None
    job.output_vtt_path = None
    job.output_original_vtt_path = None
    job.processing_time_seconds = None
    job.transcription_data = None
    job.speakers_detected = 0

    from app.workers.pipeline import run_dubbing_pipeline
    task = run_dubbing_pipeline.apply_async(args=[job_id], queue="dubbing")
    job.celery_task_id = task.id

    await db.commit()
    await db.refresh(job)
    return _job_to_response(job)


@router.delete("/{job_id}")
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.celery_task_id:
        from app.core.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    job.status = JobStatus.CANCELLED
    await db.commit()
    return {"message": "Job cancelled"}


@router.get("/{job_id}/download/video")
async def download_video(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job or job.status != JobStatus.COMPLETED:
        raise HTTPException(404, "Output not ready")
    if not job.output_video_path or not os.path.exists(job.output_video_path):
        raise HTTPException(404, "Video file not found")
    return FileResponse(
        job.output_video_path,
        media_type="video/mp4",
        filename=Path(job.output_video_path).name,
    )


@router.get("/{job_id}/download/srt")
async def download_srt(job_id: str, lang: str = "translated", db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job or job.status != JobStatus.COMPLETED:
        raise HTTPException(404, "Output not ready")

    path = job.output_srt_path if lang == "translated" else job.output_original_srt_path
    if not path or not os.path.exists(path):
        raise HTTPException(404, "SRT file not found")

    return FileResponse(
        path,
        media_type="text/plain",
        filename=Path(path).name,
    )


@router.get("/{job_id}/download/vtt")
async def download_vtt(job_id: str, lang: str = "translated", db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job or job.status != JobStatus.COMPLETED:
        raise HTTPException(404, "Output not ready")

    path = job.output_vtt_path if lang == "translated" else job.output_original_vtt_path
    if not path or not os.path.exists(path):
        raise HTTPException(404, "VTT file not found")

    return FileResponse(
        path,
        media_type="text/vtt",
        filename=Path(path).name,
    )


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status,
        progress=job.progress or 0.0,
        current_stage=job.current_stage or "",
        original_filename=job.original_filename,
        source_language=job.source_language,
        target_language=job.target_language,
        speakers_detected=job.speakers_detected or 0,
        duration_seconds=job.duration_seconds,
        output_video_path=job.output_video_path,
        output_srt_path=job.output_srt_path,
        output_original_srt_path=job.output_original_srt_path,
        output_vtt_path=getattr(job, "output_vtt_path", None),
        output_original_vtt_path=getattr(job, "output_original_vtt_path", None),
        error_message=job.error_message,
        processing_time_seconds=job.processing_time_seconds,
        created_at=job.created_at.isoformat() if job.created_at else "",
        transcription_data=job.transcription_data,
    )
