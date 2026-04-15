from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    app_name: str = "DubbingStudio"
    debug: bool = False

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    uploads_dir: Path = data_dir / "uploads"
    outputs_dir: Path = data_dir / "outputs"
    temp_dir: Path = data_dir / "temp"
    models_dir: Path = data_dir / "models"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/dubbing_studio.db"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # GPU configuration
    primary_gpu_id: int = 0
    secondary_gpu_id: int = 1
    use_gpu: bool = True

    # WhisperX settings (replaces standalone faster-whisper + pyannote)
    whisper_model_size: str = "large-v3-turbo"
    whisper_batch_size: int = 16
    whisper_compute_type: str = "float16"

    # Fish Speech settings (replaces Coqui XTTS)
    fish_speech_model: str = "fishaudio/fish-speech-1.5"
    fish_speech_device: str = "cuda"

    # Translation — MADLAD-400 (replaces NLLB-200, Apache 2.0 licensed)
    translation_model: str = "google/madlad400-3b-mt"
    translation_device: str = "cuda"

    # Speaker diarization (built into WhisperX via pyannote community-1)
    hf_token: Optional[str] = None

    # CORS / server
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_dir: str = "../frontend/dist"

    # Supported languages (ISO 639-1)
    # MADLAD-400 uses <2xx> prefix tags (ISO 639-1)
    # WhisperX uses Whisper language codes
    # Fish Speech supports 80+ languages natively
    supported_languages: dict = {
        "en": {"name": "English", "madlad": "en", "whisper": "en"},
        "es": {"name": "Spanish", "madlad": "es", "whisper": "es"},
        "fr": {"name": "French", "madlad": "fr", "whisper": "fr"},
        "de": {"name": "German", "madlad": "de", "whisper": "de"},
        "it": {"name": "Italian", "madlad": "it", "whisper": "it"},
        "pt": {"name": "Portuguese", "madlad": "pt", "whisper": "pt"},
        "ja": {"name": "Japanese", "madlad": "ja", "whisper": "ja"},
        "zh": {"name": "Chinese", "madlad": "zh", "whisper": "zh"},
        "ko": {"name": "Korean", "madlad": "ko", "whisper": "ko"},
        "ar": {"name": "Arabic", "madlad": "ar", "whisper": "ar"},
        "ru": {"name": "Russian", "madlad": "ru", "whisper": "ru"},
        "hi": {"name": "Hindi", "madlad": "hi", "whisper": "hi"},
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

for d in [settings.uploads_dir, settings.outputs_dir, settings.temp_dir, settings.models_dir]:
    d.mkdir(parents=True, exist_ok=True)
