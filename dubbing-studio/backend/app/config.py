from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_DATA_DIR = _PROJECT_ROOT / "data"


class Settings(BaseSettings):
    app_name: str = "DubbingStudio"
    debug: bool = False

    base_dir: Path = _PROJECT_ROOT
    data_dir: Path = _DATA_DIR
    uploads_dir: Path = _DATA_DIR / "uploads"
    outputs_dir: Path = _DATA_DIR / "outputs"
    temp_dir: Path = _DATA_DIR / "temp"
    models_dir: Path = _DATA_DIR / "models"

    database_url: str = f"sqlite+aiosqlite:///{_DATA_DIR / 'dubbing_studio.db'}"

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

    # Translation — NLLB-200 3.3B (Meta, better translation quality)
    translation_model: str = "facebook/nllb-200-3.3B"
    translation_device: str = "cuda"

    # Speaker diarization (built into WhisperX via pyannote community-1)
    hf_token: Optional[str] = None

    # CORS / server
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_dir: str = "../frontend/dist"

    # Supported languages (ISO 639-1)
    # NLLB-200 uses BCP-47 style codes with script (e.g. spa_Latn)
    # WhisperX uses Whisper language codes
    # Fish Speech supports 80+ languages natively
    supported_languages: dict = {
        "en": {"name": "English", "nllb": "eng_Latn", "whisper": "en"},
        "es": {"name": "Spanish", "nllb": "spa_Latn", "whisper": "es"},
        "fr": {"name": "French", "nllb": "fra_Latn", "whisper": "fr"},
        "de": {"name": "German", "nllb": "deu_Latn", "whisper": "de"},
        "it": {"name": "Italian", "nllb": "ita_Latn", "whisper": "it"},
        "pt": {"name": "Portuguese", "nllb": "por_Latn", "whisper": "pt"},
        "ja": {"name": "Japanese", "nllb": "jpn_Jpan", "whisper": "ja"},
        "zh": {"name": "Chinese", "nllb": "zho_Hans", "whisper": "zh"},
        "ko": {"name": "Korean", "nllb": "kor_Hang", "whisper": "ko"},
        "ar": {"name": "Arabic", "nllb": "arb_Arab", "whisper": "ar"},
        "ru": {"name": "Russian", "nllb": "rus_Cyrl", "whisper": "ru"},
        "hi": {"name": "Hindi", "nllb": "hin_Deva", "whisper": "hi"},
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

for d in [settings.uploads_dir, settings.outputs_dir, settings.temp_dir, settings.models_dir]:
    d.mkdir(parents=True, exist_ok=True)
