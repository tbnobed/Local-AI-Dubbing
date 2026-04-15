"""
Transcription + word alignment + speaker diarization using WhisperX.

WhisperX combines:
  - faster-whisper (CTranslate2) for fast transcription
  - wav2vec2 forced alignment for word-level timestamps
  - pyannote community-1 for speaker diarization

This replaces the separate faster-whisper + pyannote pipeline.
"""
import gc
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    start: float
    end: float
    word: str
    score: float = 1.0
    speaker: Optional[str] = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[WordSegment] = field(default_factory=list)
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    language: str
    segments: list[Segment]
    duration: float
    num_speakers: int


class TranscriptionService:
    """
    All-in-one service: transcribe → align → diarize using WhisperX.
    """

    def __init__(self, config):
        self.config = config

    def transcribe_and_diarize(
        self,
        audio_path: str,
        source_language: Optional[str] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        import whisperx
        import torch

        device = f"cuda:{self.config.primary_gpu_id}" if self.config.use_gpu else "cpu"
        hf_token = self.config.hf_token

        # --- Stage 1: Transcribe ---
        if progress_callback:
            progress_callback(0.0, "transcribing")

        logger.info(f"Loading WhisperX model '{self.config.whisper_model_size}' on {device}")
        model = whisperx.load_model(
            self.config.whisper_model_size,
            device,
            compute_type=self.config.whisper_compute_type,
            download_root=str(self.config.models_dir / "whisperx"),
        )

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        logger.info("Transcribing...")
        result = model.transcribe(
            audio,
            batch_size=self.config.whisper_batch_size,
            language=source_language if source_language else None,
        )
        detected_lang = result.get("language", source_language or "en")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(0.35, "aligning")

        # --- Stage 2: Word-level alignment ---
        logger.info(f"Aligning words (lang={detected_lang})...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(0.55, "diarizing")

        # --- Stage 3: Speaker diarization ---
        num_speakers = 1
        if hf_token:
            try:
                logger.info("Running speaker diarization (pyannote community-1)...")
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device,
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                speakers = set()
                for seg in result["segments"]:
                    if "speaker" in seg:
                        speakers.add(seg["speaker"])
                num_speakers = len(speakers) if speakers else 1

                del diarize_model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Diarization failed (continuing without it): {e}")
        else:
            logger.warning("HF_TOKEN not set — skipping speaker diarization")

        if progress_callback:
            progress_callback(0.9, "finalizing")

        # --- Build structured output ---
        segments = []
        for i, seg in enumerate(result["segments"]):
            words = []
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    words.append(WordSegment(
                        start=w["start"],
                        end=w["end"],
                        word=w.get("word", ""),
                        score=w.get("score", 1.0),
                        speaker=w.get("speaker"),
                    ))

            segments.append(Segment(
                id=i,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                words=words,
                speaker=seg.get("speaker", "SPEAKER_00"),
            ))

        logger.info(f"WhisperX complete: {len(segments)} segments, {num_speakers} speakers, lang={detected_lang}")

        if progress_callback:
            progress_callback(1.0, "done")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=duration,
            num_speakers=num_speakers,
        )
