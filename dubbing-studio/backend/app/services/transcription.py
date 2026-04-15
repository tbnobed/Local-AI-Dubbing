"""
Transcription + word alignment + speaker diarization.

Uses pure PyTorch Whisper (via transformers) for transcription — this works on
ALL GPUs including Blackwell sm_120, unlike CTranslate2/faster-whisper.

Then uses WhisperX for:
  - wav2vec2 forced alignment (word-level timestamps)
  - pyannote community-1 speaker diarization
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
      2. WhisperX alignment (wav2vec2)
      3. WhisperX diarization (pyannote)
    """

    def __init__(self, config):
        self.config = config

    def _get_device(self) -> str:
        if self.config.use_gpu and torch.cuda.is_available():
            return f"cuda:{self.config.primary_gpu_id}"
        return "cpu"

    def _transcribe_with_transformers(
        self, audio_path: str, source_language: Optional[str] = None
    ) -> tuple[list[dict], str]:
        """Transcribe using HuggingFace transformers Whisper (pure PyTorch)."""
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

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=self.config.whisper_batch_size,
            return_timestamps=True,
        )

        logger.info("Transcribing with transformers Whisper...")
        result = pipe(audio_path, generate_kwargs=generate_kwargs)

        detected_lang = source_language or "en"

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

        del pipe, model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Transcription complete: {len(segments)} raw segments, lang={detected_lang}")
        return segments, detected_lang

    def transcribe_and_diarize(
        self,
        audio_path: str,
        source_language: Optional[str] = None,
        progress_callback=None,
    ) -> TranscriptionResult:
        import whisperx

        device = self._get_device()
        hf_token = self.config.hf_token

        # --- Stage 1: Transcribe (pure PyTorch — works on Blackwell) ---
        if progress_callback:
            progress_callback(0.0, "transcribing")

        raw_segments, detected_lang = self._transcribe_with_transformers(
            audio_path, source_language
        )

        if progress_callback:
            progress_callback(0.35, "aligning")

        # --- Stage 2: Word-level alignment (WhisperX / wav2vec2) ---
        # Use the secondary GPU for alignment to keep primary GPU free for
        # later pipeline stages (translation, TTS). Falls back to CPU if it crashes.
        if self.config.use_gpu and torch.cuda.device_count() > 1:
            align_device = f"cuda:{self.config.secondary_gpu_id}"
        elif self.config.use_gpu and torch.cuda.is_available():
            align_device = f"cuda:{self.config.primary_gpu_id}"
        else:
            align_device = "cpu"
        logger.info(f"Aligning words with WhisperX (lang={detected_lang}) on {align_device}...")
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0

        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=align_device,
            )
            aligned = whisperx.align(
                raw_segments,
                model_a,
                metadata,
                audio,
                align_device,
                return_char_alignments=False,
            )

            del model_a
            gc.collect()

            aligned_segments = aligned.get("segments", raw_segments)
        except Exception as e:
            logger.warning(f"Alignment failed (using raw timestamps): {e}")
            aligned_segments = raw_segments

        if progress_callback:
            progress_callback(0.55, "diarizing")

        # --- Stage 3: Speaker diarization (pyannote via WhisperX) ---
        num_speakers = 1
        result_data = {"segments": aligned_segments}

        if hf_token:
            try:
                # Use secondary GPU for diarization (keeps primary GPU free for TTS/translation)
                if self.config.use_gpu and torch.cuda.device_count() > 1:
                    diarize_device = torch.device(f"cuda:{self.config.secondary_gpu_id}")
                elif self.config.use_gpu and torch.cuda.is_available():
                    diarize_device = torch.device(f"cuda:{self.config.primary_gpu_id}")
                else:
                    diarize_device = torch.device("cpu")
                logger.info(f"Running speaker diarization (pyannote) on {diarize_device}...")

                # Try whisperx.DiarizationPipeline first, fall back to pyannote directly
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

                    try:
                        result_data = whisperx.assign_word_speakers(diarize_segments, result_data)
                    except (AttributeError, Exception) as assign_err:
                        logger.warning(f"Speaker assignment failed: {assign_err}")
                        # Manual speaker assignment from diarization turns
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
