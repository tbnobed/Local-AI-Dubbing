from celery import Celery
from app.config import settings


celery_app = Celery(
    "dubbing_studio",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.pipeline"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.workers.pipeline.run_dubbing_pipeline": {"queue": "dubbing"},
    },
    task_soft_time_limit=7200,
    task_time_limit=10800,
)
