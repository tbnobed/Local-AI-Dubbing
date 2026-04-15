"""
System status API - GPU info, model status, system health.
"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/system", tags=["system"])
logger = logging.getLogger(__name__)


class GPUInfo(BaseModel):
    id: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature_c: Optional[float]


class SystemStatus(BaseModel):
    gpus: list[GPUInfo]
    redis_connected: bool
    celery_workers: int
    supported_languages: dict


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    from app.config import settings

    gpus = []
    try:
        import GPUtil
        detected = GPUtil.getGPUs()
        for g in detected:
            gpus.append(GPUInfo(
                id=g.id,
                name=g.name,
                memory_total_mb=g.memoryTotal,
                memory_used_mb=g.memoryUsed,
                memory_free_mb=g.memoryFree,
                utilization_percent=g.load * 100,
                temperature_c=g.temperature,
            ))
    except Exception as e:
        logger.warning(f"GPUtil not available: {e}")

    redis_ok = False
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        redis_ok = True
        r.close()
    except Exception:
        pass

    celery_workers = 0
    try:
        from app.core.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=1.0)
        active = inspect.active()
        if active:
            celery_workers = len(active)
    except Exception:
        pass

    return SystemStatus(
        gpus=gpus,
        redis_connected=redis_ok,
        celery_workers=celery_workers,
        supported_languages=settings.supported_languages,
    )


@router.get("/languages")
async def get_supported_languages():
    from app.config import settings
    return {
        code: {"name": info["name"], "code": code}
        for code, info in settings.supported_languages.items()
    }
