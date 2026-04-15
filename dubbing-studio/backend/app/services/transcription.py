"""
Transcription service using faster-whisper with CUDA acceleration.
Produces word-level timestamps for precise dubbing alignment.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[WordSegment] = field(default_factory=list)
    speaker: Optional[str] = None
    avg_logprob: float = 0.0


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    segments: list[Segment]
    duration: float


class TranscriptionService:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._model_size = config.whisper_model_size
        self._device = config.whisper_device
        self._compute_type = config.whisper_compute_type

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model '{self._model_size}' on {self._device}...")
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=str(self.config.models_dir / "whisper"),
            )
            logger.info("Whisper model loaded.")
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        model = self._load_model()

        logger.info(f"Transcribing: {audio_path}, language hint: {language}")

        segments_raw, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )

        segments = []
        total_duration = info.duration if info.duration else 1.0

        for i, seg in enumerate(segments_raw):
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(WordSegment(
                        start=w.start,
                        end=w.end,
                        word=w.word,
                        probability=w.probability,
                    ))

            segments.append(Segment(
                id=i,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
                avg_logprob=seg.avg_logprob,
            ))

            if progress_callback:
                progress_callback(seg.end / total_duration)

        logger.info(f"Transcription complete: {len(segments)} segments, language={info.language}")

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=segments,
            duration=info.duration,
        )

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded.")
