"""
Celery worker - orchestrates the full dubbing pipeline.
Stages: audio extraction → transcription → diarization → translation → TTS → mixing → output
"""
import logging
import os
import time
from pathlib import Path
from datetime import datetime

from app.core.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)

STAGE_WEIGHTS = {
    "extracting_audio": 0.05,
    "transcribing": 0.20,
    "diarizing": 0.10,
    "translating": 0.15,
    "cloning_voices": 0.05,
    "synthesizing": 0.35,
    "mixing": 0.10,
}


def update_job_sync(session_factory, job_id: str, **kwargs):
    """Synchronous DB update from Celery worker."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.job import Job

    sync_url = settings.database_url.replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        job = session.get(Job, job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = datetime.utcnow()
            session.commit()


def push_ws_update(job_id: str, data: dict):
    """Push WebSocket update from worker context via Redis pub/sub."""
    import redis
    import json

    try:
        r = redis.from_url(settings.redis_url)
        r.publish(f"job:{job_id}", json.dumps(data))
    except Exception as e:
        logger.warning(f"Failed to push WS update: {e}")


@celery_app.task(bind=True, name="app.workers.pipeline.run_dubbing_pipeline")
def run_dubbing_pipeline(self, job_id: str):
    start_time = time.time()
    logger.info(f"Starting pipeline for job {job_id}")

    temp_dir = settings.temp_dir / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    def update_progress(stage: str, stage_progress: float = 0.0):
        base = sum(v for k, v in STAGE_WEIGHTS.items()
                   if list(STAGE_WEIGHTS.keys()).index(k) < list(STAGE_WEIGHTS.keys()).index(stage))
        weight = STAGE_WEIGHTS.get(stage, 0.0)
        total_progress = (base + weight * stage_progress) * 100

        update_job_sync(None, job_id,
                        status=stage,
                        progress=total_progress,
                        current_stage=stage)

        push_ws_update(job_id, {
            "job_id": job_id,
            "status": stage,
            "progress": total_progress,
            "stage": stage,
        })

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models.job import Job, JobStatus

        sync_url = settings.database_url.replace("+aiosqlite", "")
        engine = create_engine(sync_url)
        Session = sessionmaker(bind=engine)

        with Session() as session:
            job = session.get(Job, job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            input_path = job.input_path
            source_language = job.source_language
            target_language = job.target_language
            voice_cloning_enabled = bool(job.voice_cloning_enabled)
            export_srt = bool(job.export_srt)
            original_filename = job.original_filename

        output_dir = settings.outputs_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Extract audio
        update_progress("extracting_audio", 0.0)
        from app.services.audio_mixer import AudioMixerService
        mixer = AudioMixerService(settings)

        audio_path = str(temp_dir / "audio.wav")
        mixer.extract_audio(input_path, audio_path)
        video_info = mixer.get_video_info(input_path)
        total_duration = video_info.get("duration", 0.0)
        update_progress("extracting_audio", 1.0)

        # Stage 2: Transcribe
        update_progress("transcribing", 0.0)
        from app.services.transcription import TranscriptionService
        transcriber = TranscriptionService(settings)

        lang_hint = settings.supported_languages.get(source_language, {}).get("whisper", None)
        transcription = transcriber.transcribe(
            audio_path,
            language=lang_hint if lang_hint != "auto" else None,
            progress_callback=lambda p: update_progress("transcribing", p),
        )
        transcriber.unload()
        segments = transcription.segments
        update_progress("transcribing", 1.0)

        # Stage 3: Speaker diarization
        update_progress("diarizing", 0.0)
        from app.services.diarization import DiarizationService
        diarizer = DiarizationService(settings)

        speakers_detected = 1
        speaker_samples = {}

        try:
            diarization = diarizer.diarize(audio_path)
            speakers_detected = diarization.num_speakers
            segments = diarizer.assign_speakers_to_segments(segments, diarization)

            samples_dir = str(temp_dir / "speaker_samples")
            speaker_samples = diarizer.extract_speaker_voice_samples(
                audio_path, diarization, samples_dir
            )
            diarizer.unload()
        except Exception as e:
            logger.warning(f"Diarization failed, using single speaker: {e}")
            for seg in segments:
                seg.speaker = "SPEAKER_00"
            speaker_samples = {"SPEAKER_00": audio_path}

        update_progress("diarizing", 1.0)

        # Stage 4: Translation
        update_progress("translating", 0.0)
        from app.services.translation import TranslationService
        translator = TranslationService(settings)

        segments = translator.translate_segments(
            segments,
            source_language,
            target_language,
            progress_callback=lambda p: update_progress("translating", p),
        )
        translator.unload()
        update_progress("translating", 1.0)

        # Stage 5: Generate SRT files
        update_progress("cloning_voices", 0.0)
        from app.services.subtitle_generator import generate_both_srts

        if export_srt:
            original_srt, translated_srt = generate_both_srts(
                segments,
                str(output_dir),
                job_id,
                source_language,
                target_language,
            )
        else:
            original_srt = None
            translated_srt = None

        # Ensure we have voice samples
        if not speaker_samples:
            speaker_samples = {"SPEAKER_00": audio_path}

        update_progress("cloning_voices", 1.0)

        # Stage 6: TTS synthesis with voice cloning
        update_progress("synthesizing", 0.0)
        synth_dir = str(temp_dir / "synthesized")

        if voice_cloning_enabled:
            from app.services.tts import TTSService
            tts = TTSService(settings)
            segments = tts.synthesize_all_segments(
                segments,
                speaker_samples,
                target_language,
                synth_dir,
                progress_callback=lambda p: update_progress("synthesizing", p),
            )
            tts.unload()
        else:
            for seg in segments:
                seg.synth_audio_path = None

        update_progress("synthesizing", 1.0)

        # Stage 7: Mix and produce final video
        update_progress("mixing", 0.0)

        dubbed_audio_path = str(temp_dir / "dubbed_audio.wav")
        mixer.build_dubbed_audio(segments, total_duration, dubbed_audio_path)

        stem = Path(original_filename).stem
        output_video_path = str(output_dir / f"{stem}_dubbed_{target_language}.mp4")
        mixer.merge_audio_into_video(input_path, dubbed_audio_path, output_video_path)

        processing_time = time.time() - start_time

        transcription_data = {
            "language": transcription.language,
            "language_probability": transcription.language_probability,
            "segments_count": len(segments),
            "duration": total_duration,
        }

        update_job_sync(None, job_id,
                        status=JobStatus.COMPLETED,
                        progress=100.0,
                        current_stage="completed",
                        output_video_path=output_video_path,
                        output_srt_path=translated_srt,
                        output_original_srt_path=original_srt,
                        duration_seconds=total_duration,
                        speakers_detected=speakers_detected,
                        transcription_data=transcription_data,
                        processing_time_seconds=processing_time)

        push_ws_update(job_id, {
            "job_id": job_id,
            "status": "completed",
            "progress": 100.0,
            "stage": "completed",
            "output_video_path": output_video_path,
        })

        logger.info(f"Pipeline complete for {job_id} in {processing_time:.1f}s")

    except Exception as e:
        logger.error(f"Pipeline failed for {job_id}: {e}", exc_info=True)
        from app.models.job import JobStatus
        update_job_sync(None, job_id,
                        status=JobStatus.FAILED,
                        progress=0.0,
                        error_message=str(e))
        push_ws_update(job_id, {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        })
        raise
