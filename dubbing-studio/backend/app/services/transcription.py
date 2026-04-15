"""
Transcription + word alignment + speaker diarization.

Uses pure PyTorch Whisper (via transformers) for transcription — this works on
ALL GPUs including Blackwell sm_120, unlike CTranslate2/faster-whisper.

Word-level timestamps come from Whisper itself (return_timestamps="word"),
eliminating the need for wav2vec2 forced alignment (which hangs on Blackwell).

Speaker diarization uses pyannote via WhisperX.
"""
import gc
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

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
    translated_text: Optional[str] = None
    synth_audio_path: Optional[str] = None


@dataclass
class TranscriptionResult:
    language: str
    segments: list[Segment]
    duration: float
    num_speakers: int


class TranscriptionService:
    """
    Transcription pipeline:
      1. Whisper via transformers (pure PyTorch — Blackwell-safe)
         - Segment-level timestamps for subtitles-only mode
         - Word-level timestamps for diarization mode
      2. Speaker diarization (pyannote)
    """

    def __init__(self, config):
        self.config = config

    def _get_device(self) -> str:
        if self.config.use_gpu and torch.cuda.is_available():
            return f"cuda:{self.config.primary_gpu_id}"
        return "cpu"

    def _transcribe_with_transformers(
        self,
        audio_path: str,
        source_language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> tuple[list[dict], str]:
        """Transcribe using HuggingFace transformers Whisper (pure PyTorch).

        Args:
            word_timestamps: If True, return word-level timestamps (for diarization).
                             If False, return segment-level timestamps (faster, for subtitles-only).
        """
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = self._get_device()
        torch_dtype = torch.float16 if "cuda" in device else torch.float32

        model_id = "openai/whisper-large-v3-turbo"
        logger.info(f"Loading transformers Whisper '{model_id}' on {device} ({torch_dtype})")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=str(self.config.models_dir / "whisper-hf"),
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(self.config.models_dir / "whisper-hf"),
        )

        generate_kwargs = {}
        if source_language:
            generate_kwargs["language"] = source_language

        ts_mode = "word" if word_timestamps else True
        batch_size = 2 if word_timestamps else self.config.whisper_batch_size

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps=ts_mode,
        )

        mode_label = "word-level" if word_timestamps else "segment-level"
        logger.info(f"Transcribing with transformers Whisper ({mode_label} timestamps)...")
        result = pipe(audio_path, generate_kwargs=generate_kwargs)

        detected_lang = source_language or "en"

        if word_timestamps:
            segments, words_flat = self._parse_word_level_output(result)
            logger.info(f"Transcription complete: {len(segments)} segments, {len(words_flat)} words, lang={detected_lang}")
        else:
            segments = self._parse_segment_level_output(result)
            logger.info(f"Transcription complete: {len(segments)} segments, lang={detected_lang}")

        del pipe, model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segments, detected_lang

    def _parse_segment_level_output(self, result: dict) -> list[dict]:
        segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp", (0.0, 0.0))
                start = ts[0] if ts[0] is not None else 0.0
                end = ts[1] if ts[1] is not None else start + 1.0
                segments.append({
                    "start": start,
                    "end": end,
                    "text": chunk.get("text", "").strip(),
                })
        elif "text" in result:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": result["text"].strip(),
            })
        return segments

    def _parse_word_level_output(self, result: dict) -> tuple[list[dict], list[dict]]:
        """Parse word-level Whisper output into segments with word arrays.

        Groups consecutive words into segments based on pauses (>0.5s gap)
        and sentence boundaries (. ! ?).
        """
        words_flat = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp", (0.0, 0.0))
                start = ts[0] if ts[0] is not None else 0.0
                end = ts[1] if ts[1] is not None else start
                word_text = chunk.get("text", "").strip()
                if word_text:
                    words_flat.append({
                        "start": start,
                        "end": end if end > start else start + 0.1,
                        "word": word_text,
                        "score": 1.0,
                    })

        if not words_flat:
            text = result.get("text", "").strip()
            return [{"start": 0.0, "end": 0.0, "text": text, "words": []}], []

        segments = []
        current_words = []
        PAUSE_THRESHOLD = 0.5
        SENTENCE_ENDINGS = {'.', '!', '?'}

        for i, w in enumerate(words_flat):
            current_words.append(w)

            is_sentence_end = any(w["word"].rstrip().endswith(c) for c in SENTENCE_ENDINGS)
            next_has_gap = (
                i + 1 < len(words_flat)
                and words_flat[i + 1]["start"] - w["end"] > PAUSE_THRESHOLD
            )
            is_last = i == len(words_flat) - 1

            if is_sentence_end or next_has_gap or is_last:
                seg_text = " ".join(cw["word"] for cw in current_words).strip()
                seg_text = seg_text.replace("  ", " ")
                segments.append({
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "text": seg_text,
                    "words": list(current_words),
                })
                current_words = []

        return segments, words_flat

    def transcribe_only(
        self,
        audio_path: str,
        source_language: Optional[str] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        """Fast path: transcription only, no alignment or diarization.
        Used for subtitle-only jobs where we just need segment timestamps.
        """
        import whisperx

        if progress_callback:
            progress_callback(0.0, "transcribing")

        raw_segments, detected_lang = self._transcribe_with_transformers(
            audio_path, source_language, word_timestamps=False
        )

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        if progress_callback:
            progress_callback(0.8, "finalizing")

        segments = []
        for i, seg in enumerate(raw_segments):
            segments.append(Segment(
                id=i,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                words=[],
                speaker="SPEAKER_00",
            ))

        logger.info(f"Transcription complete (fast): {len(segments)} segments, lang={detected_lang}")

        if progress_callback:
            progress_callback(1.0, "done")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=duration,
            num_speakers=0,
        )

    def transcribe_and_diarize(
        self,
        audio_path: str,
        source_language: Optional[str] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        import whisperx

        hf_token = self.config.hf_token

        # --- Stage 1: Transcribe with word-level timestamps (all on GPU 0) ---
        if progress_callback:
            progress_callback(0.0, "transcribing")

        raw_segments, detected_lang = self._transcribe_with_transformers(
            audio_path, source_language, word_timestamps=True
        )

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        if progress_callback:
            progress_callback(0.50, "diarizing")

        # --- Stage 2: Speaker diarization (pyannote) ---
        num_speakers = 1
        result_data = {"segments": raw_segments}

        if hf_token:
            try:
                # pyannote hangs on Blackwell (sm_120) GPUs — use CPU
                # Diarization is fast on CPU; the bottleneck was alignment (now handled by Whisper)
                diarize_device = torch.device("cpu")
                logger.info(f"Running speaker diarization (pyannote) on CPU...")

                diarize_model = None
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=hf_token,
                        device=diarize_device,
                    )
                except (AttributeError, TypeError):
                    logger.info("whisperx.DiarizationPipeline not available, using pyannote directly")
                    from pyannote.audio import Pipeline as PyannotePipeline
                    try:
                        diarize_model = PyannotePipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            token=hf_token,
                        ).to(diarize_device)
                    except TypeError:
                        diarize_model = PyannotePipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token,
                        ).to(diarize_device)

                if diarize_model is not None:
                    if hasattr(diarize_model, '__call__'):
                        diarize_segments = diarize_model(audio_path)
                    else:
                        diarize_segments = diarize_model(audio)

                    import pandas as pd
                    if not isinstance(diarize_segments, pd.DataFrame):
                        try:
                            rows = []
                            annotation = None
                            if hasattr(diarize_segments, "speaker_diarization"):
                                annotation = diarize_segments.speaker_diarization
                                logger.info(f"Extracted speaker_diarization (type: {type(annotation).__name__})")
                            elif hasattr(diarize_segments, "itertracks"):
                                annotation = diarize_segments
                            elif hasattr(diarize_segments, "to_annotation"):
                                annotation = diarize_segments.to_annotation()

                            if annotation is not None and hasattr(annotation, "itertracks"):
                                for turn, _, speaker in annotation.itertracks(yield_label=True):
                                    rows.append({
                                        "start": turn.start,
                                        "end": turn.end,
                                        "speaker": speaker,
                                    })

                            if rows:
                                diarize_segments = pd.DataFrame(rows)
                                logger.info(f"Converted diarization: {len(rows)} turns, "
                                            f"{diarize_segments['speaker'].nunique()} speakers")
                            else:
                                logger.warning(f"Diarization returned empty result "
                                               f"(type: {type(diarize_segments).__name__}, "
                                               f"attrs: {[a for a in dir(diarize_segments) if not a.startswith('_')]})")
                                diarize_segments = pd.DataFrame()
                        except Exception as conv_err:
                            logger.warning(f"Could not convert diarization output: {conv_err}")
                            diarize_segments = pd.DataFrame()

                    try:
                        if not diarize_segments.empty:
                            result_data = whisperx.assign_word_speakers(diarize_segments, result_data)
                        else:
                            for seg in result_data["segments"]:
                                seg["speaker"] = "SPEAKER_00"
                    except Exception as assign_err:
                        logger.warning(f"Speaker assignment failed: {assign_err}")
                        for seg in result_data["segments"]:
                            seg["speaker"] = "SPEAKER_00"

                    speakers = set()
                    for seg in result_data["segments"]:
                        if "speaker" in seg:
                            speakers.add(seg["speaker"])
                    num_speakers = len(speakers) if speakers else 1

                    del diarize_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Diarization failed (continuing without it): {e}")
        else:
            logger.warning("HF_TOKEN not set — skipping speaker diarization")

        if progress_callback:
            progress_callback(0.9, "finalizing")

        # --- Build structured output ---
        segments = []
        for i, seg in enumerate(result_data["segments"]):
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

        logger.info(f"Pipeline complete: {len(segments)} segments, {num_speakers} speakers, lang={detected_lang}")

        if progress_callback:
            progress_callback(1.0, "done")

        return TranscriptionResult(
            language=detected_lang,
            segments=segments,
            duration=duration,
            num_speakers=num_speakers,
        )
