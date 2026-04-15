from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text, Enum as SAEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid


Base = declarative_base()


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    TRANSLATING = "translating"
    CLONING_VOICES = "cloning_voices"
    SYNTHESIZING = "synthesizing"
    MIXING = "mixing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Input
    original_filename = Column(String, nullable=False)
    input_path = Column(String, nullable=False)
    source_language = Column(String, nullable=False)
    target_language = Column(String, nullable=False)

    # Options
    voice_cloning_enabled = Column(Integer, default=1)
    export_srt = Column(Integer, default=1)

    # Status
    status = Column(String, default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)
    current_stage = Column(String, default="")
    error_message = Column(Text, nullable=True)

    # Results
    output_video_path = Column(String, nullable=True)
    output_srt_path = Column(String, nullable=True)
    output_original_srt_path = Column(String, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Metadata
    transcription_data = Column(JSON, nullable=True)
    speakers_detected = Column(Integer, default=0)
    celery_task_id = Column(String, nullable=True)

    # Stats
    processing_time_seconds = Column(Float, nullable=True)
    gpu_used = Column(String, nullable=True)
