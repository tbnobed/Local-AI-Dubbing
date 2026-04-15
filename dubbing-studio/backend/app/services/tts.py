"""
Text-to-Speech with voice cloning using Fish Speech 1.5.

Fish Speech advantages over Coqui XTTS v2:
  - Actively maintained (XTTS/Coqui is discontinued)
  - State-of-the-art zero-shot voice cloning quality
  - 80+ languages supported
  - Python 3.12 native, CUDA 12.x compatible
  - ~15x real-time on RTX 4090 class GPUs

Uses a Dual-AR architecture (4B slow + 400M fast) for high-fidelity synthesis.
"""
import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self, config):
        self.config = config
        self._tts = None
        self._sample_rate = None

    def _load_model(self):
        if self._tts is None:
            try:
                from fish_speech_lib.inference import FishSpeech
                logger.info("Loading Fish Speech model via fish-speech-lib...")
                device = f"cuda:{self.config.primary_gpu_id}" if self.config.use_gpu else "cpu"
                self._tts = FishSpeech(device=device)
                self._sample_rate = 44100
                logger.info("Fish Speech loaded via fish-speech-lib.")
            except ImportError:
                logger.info("fish-speech-lib not available, trying fish-speech-rs...")
                try:
                    from fish_speech import FireflyCodec, LM
                    from huggingface_hub import snapshot_download

                    model_dir = snapshot_download(
                        self.config.fish_speech_model,
                        cache_dir=str(self.config.models_dir / "fish-speech"),
                    )
                    codec = FireflyCodec(model_dir, version="1.5", device="cuda")
                    lm = LM(model_dir, version="1.5", device="cuda", dtype="bf16")
                    self._tts = {"codec": codec, "lm": lm, "type": "native"}
                    self._sample_rate = codec.sample_rate
                    logger.info("Fish Speech loaded via fish-speech-rs.")
                except ImportError:
                    logger.error(
                        "Neither fish-speech-lib nor fish-speech-rs found. "
                        "Install one: pip install fish-speech-lib OR pip install fish-speech-rs"
                    )
                    raise RuntimeError("No Fish Speech backend available")

        return self._tts

    def synthesize_segment(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        output_path: str,
    ) -> float:
        """
        Synthesize speech for one segment, cloning the provided speaker voice.
        Returns duration of synthesized audio in seconds.
        """
        tts = self._load_model()

        if isinstance(tts, dict) and tts.get("type") == "native":
            # fish-speech-rs native API
            codec = tts["codec"]
            lm = tts["lm"]

            ref_audio, ref_sr = sf.read(speaker_wav, dtype="float32")
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.reshape(1, -1)

            codes = codec.encode(ref_audio, ref_sr)
            speaker_prompt = lm.get_speaker_prompt(
                [{"text": "", "codes": codes}],
                sysprompt="Speak out the provided text.",
            )
            generated_codes = lm.generate([text], speaker_prompt=speaker_prompt)
            pcm = codec.decode(generated_codes)

            sf.write(output_path, pcm.flatten(), self._sample_rate)
        else:
            # fish-speech-lib simple API
            # Read reference text from the first few seconds (empty is OK for zero-shot)
            sample_rate, audio = tts(
                text=text,
                reference_audio=speaker_wav,
                reference_audio_text="",
                max_new_tokens=2048,
                chunk_length=1000,
            )
            sf.write(output_path, audio, sample_rate, format="WAV")
            self._sample_rate = sample_rate

        info = sf.info(output_path)
        return info.duration

    def time_stretch_audio(
        self,
        audio_path: str,
        output_path: str,
        target_duration: float,
        min_rate: float = 0.75,
        max_rate: float = 1.5,
    ) -> str:
        """
        Time-stretch audio to fit within target duration using librosa.
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=None)
        current_duration = len(y) / sr

        if current_duration == 0:
            sf.write(output_path, y, sr)
            return output_path

        rate = current_duration / target_duration
        rate = max(min_rate, min(max_rate, rate))

        if abs(rate - 1.0) < 0.02:
            sf.write(output_path, y, sr)
            return output_path

        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        sf.write(output_path, y_stretched, sr)
        return output_path

    def synthesize_all_segments(
        self,
        segments: list,
        speaker_samples: dict[str, str],
        target_language: str,
        output_dir: str,
        progress_callback=None,
    ) -> list:
        """
        Synthesize all translated segments with voice cloning.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        default_speaker = list(speaker_samples.values())[0] if speaker_samples else None

        total = len(segments)
        for i, seg in enumerate(segments):
            translated_text = getattr(seg, "translated_text", seg.text)
            if not translated_text.strip():
                seg.synth_audio_path = None
                seg.synth_duration = 0.0
                continue

            speaker_id = getattr(seg, "speaker", "SPEAKER_00")
            speaker_wav = speaker_samples.get(speaker_id, default_speaker)

            if not speaker_wav:
                logger.warning(f"No voice sample for speaker {speaker_id}, skipping segment {i}")
                seg.synth_audio_path = None
                seg.synth_duration = 0.0
                continue

            raw_path = str(output_dir / f"seg_{i:04d}_raw.wav")
            stretched_path = str(output_dir / f"seg_{i:04d}.wav")

            try:
                synth_duration = self.synthesize_segment(
                    text=translated_text,
                    speaker_wav=speaker_wav,
                    language=target_language,
                    output_path=raw_path,
                )

                original_duration = seg.end - seg.start

                self.time_stretch_audio(
                    audio_path=raw_path,
                    output_path=stretched_path,
                    target_duration=original_duration,
                )

                seg.synth_audio_path = stretched_path
                seg.synth_duration = synth_duration

            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")
                seg.synth_audio_path = None
                seg.synth_duration = 0.0

            if progress_callback:
                progress_callback((i + 1) / total)

        return segments

    def unload(self):
        if self._tts is not None:
            del self._tts
            self._tts = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("TTS model unloaded.")
