"""
Celery worker - orchestrates the full dubbing pipeline.

Pipeline stages:
  1. Audio extraction (ffmpeg)
  2. Transcription + alignment + diarization (WhisperX — all-in-one)
  3. Translation (MADLAD-400 3B)
  4. SRT subtitle export
  --- below only when voice cloning is enabled ---
  5. Voice sample extraction + TTS (Fish Speech 1.5)
  6. Audio mixing + final video (ffmpeg)
"""
import logging
import time
from pathlib import Path
from datetime import datetime

from app.core.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)

STAGE_WEIGHTS_SUBTITLES = {
    "extracting_audio": 0.05,
    "transcribing": 0.55,
    "translating": 0.30,
    "exporting_subtitles": 0.10,
}

STAGE_WEIGHTS_FULL = {
    "extracting_audio": 0.05,
    "transcribing": 0.30,
    "translating": 0.15,
    "cloning_voices": 0.05,
    "synthesizing": 0.35,
    "mixing": 0.10,
}


def update_job_sync(job_id: str, **kwargs):
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

    stage_weights = STAGE_WEIGHTS_FULL if voice_cloning_enabled else STAGE_WEIGHTS_SUBTITLES
    stage_keys = list(stage_weights.keys())

    def update_progress(stage: str, stage_progress: float = 0.0):
        idx = stage_keys.index(stage) if stage in stage_keys else 0
        base = sum(stage_weights[k] for k in stage_keys[:idx])
        weight = stage_weights.get(stage, 0.0)
        total_progress = (base + weight * stage_progress) * 100

        update_job_sync(job_id,
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
        output_dir = settings.outputs_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage 1: Extract audio ──
        update_progress("extracting_audio", 0.0)
        from app.services.audio_mixer import AudioMixerService
        mixer = AudioMixerService(settings)

        audio_path = str(temp_dir / "audio.wav")
        mixer.extract_audio(input_path, audio_path)
        video_info = mixer.get_video_info(input_path)
        total_duration = video_info.get("duration", 0.0)
        update_progress("extracting_audio", 1.0)

        # ── Stage 2: Transcribe (+ align + diarize if voice cloning) ──
        update_progress("transcribing", 0.0)
        from app.services.transcription import TranscriptionService
        transcriber = TranscriptionService(settings)

        whisper_lang = settings.supported_languages.get(source_language, {}).get("whisper")

        def transcription_progress(pct, sub_stage):
            update_progress("transcribing", pct)

        if voice_cloning_enabled:
            transcription = transcriber.transcribe_and_diarize(
                audio_path,
                source_language=whisper_lang,
                progress_callback=transcription_progress,
            )
        else:
            transcription = transcriber.transcribe_only(
                audio_path,
                source_language=whisper_lang,
                progress_callback=transcription_progress,
            )

        segments = transcription.segments
        speakers_detected = transcription.num_speakers

        update_progress("transcribing", 1.0)

        del transcriber
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Stage 3: Translate ──
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

        # ── Stage 4: Generate SRT files ──
        original_srt = None
        translated_srt = None

        from app.services.subtitle_generator import generate_both_srts
        if export_srt:
            if voice_cloning_enabled:
                update_progress("cloning_voices", 0.0)
            else:
                update_progress("exporting_subtitles", 0.0)

            original_srt, translated_srt = generate_both_srts(
                segments, str(output_dir), job_id, source_language, target_language,
            )
            logger.info(f"SRT files generated: original={original_srt}, translated={translated_srt}")

        if not voice_cloning_enabled:
            update_progress("exporting_subtitles", 1.0)

            processing_time = time.time() - start_time

            transcription_data = {
                "language": transcription.language,
                "segments_count": len(segments),
                "duration": total_duration,
            }

            update_job_sync(job_id,
                            status=JobStatus.COMPLETED,
                            progress=100.0,
                            current_stage="completed",
                            output_video_path=None,
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
            })

            logger.info(f"Subtitles-only pipeline complete for {job_id} in {processing_time:.1f}s")
            return

        # ── Stage 5: Voice cloning + TTS (only when enabled) ──
        from app.services.diarization import extract_speaker_voice_samples
        samples_dir = str(temp_dir / "speaker_samples")
        speaker_samples = extract_speaker_voice_samples(
            audio_path, segments, samples_dir
        )

        if not speaker_samples:
            speaker_samples = {"SPEAKER_00": audio_path}

        update_progress("cloning_voices", 1.0)

        update_progress("synthesizing", 0.0)
        synth_dir = str(temp_dir / "synthesized")

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
        update_progress("synthesizing", 1.0)

        # ── Stage 6: Mix and produce final video ──
        update_progress("mixing", 0.0)

        dubbed_audio_path = str(temp_dir / "dubbed_audio.wav")
        mixer.build_dubbed_audio(segments, total_duration, dubbed_audio_path)

        stem = Path(original_filename).stem
        output_video_path = str(output_dir / f"{stem}_dubbed_{target_language}.mp4")
        mixer.merge_audio_into_video(input_path, dubbed_audio_path, output_video_path)

        processing_time = time.time() - start_time

        transcription_data = {
            "language": transcription.language,
            "segments_count": len(segments),
            "duration": total_duration,
        }

        update_job_sync(job_id,
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

        logger.info(f"Full dubbing pipeline complete for {job_id} in {processing_time:.1f}s")

    except Exception as e:
        logger.error(f"Pipeline failed for {job_id}: {e}", exc_info=True)
        from app.models.job import JobStatus
        update_job_sync(job_id,
                        status=JobStatus.FAILED,
                        progress=0.0,
                        error_message=str(e))
        push_ws_update(job_id, {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        })
        raise
