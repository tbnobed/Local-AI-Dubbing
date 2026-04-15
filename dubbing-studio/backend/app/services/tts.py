"""
Text-to-Speech with voice cloning using Coqui XTTS v2.
Clones each detected speaker's voice and synthesizes translated speech.
"""
import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class TTSService:
    def __init__(self, config):
        self.config = config
        self._tts = None

    def _load_model(self):
        if self._tts is None:
            from TTS.api import TTS
            import torch

            logger.info(f"Loading XTTS v2 model...")
            gpu = torch.cuda.is_available() and self.config.use_gpu

            os.environ["COQUI_TOS_AGREED"] = "1"
            self._tts = TTS(
                model_name=self.config.xtts_model,
                progress_bar=False,
                gpu=gpu,
            )
            logger.info("XTTS v2 loaded.")

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

        lang_map = {
            "es": "es",
            "fr": "fr",
            "en": "en",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ja": "ja",
            "zh": "zh-cn",
            "ko": "ko",
            "ar": "ar",
            "ru": "ru",
            "hi": "hi",
        }
        xtts_lang = lang_map.get(language, language)

        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=xtts_lang,
            file_path=output_path,
            speed=1.0,
        )

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
        Adds 'synth_audio_path' and 'synth_duration' to each segment.
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
