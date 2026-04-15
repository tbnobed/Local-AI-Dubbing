"""
Text-to-Speech with voice cloning using Fish Speech.

Supports Fish Speech 1.5 and 2.0 APIs, with CLI fallback.

Three-stage pipeline:
  1. Encode reference audio → VQ tokens (voice characteristics)
  2. Generate semantic tokens from text (using voice prompt)
  3. Decode tokens → waveform
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self, config):
        self.config = config
        self._engine = None
        self._sample_rate = 44100
        self._checkpoint_path = None

    def _get_fish_speech_dir(self) -> Path:
        """Locate the fish-speech repo directory."""
        candidates = [
            Path(self.config.base_dir) / "fish-speech",
            Path(self.config.base_dir).parent / "fish-speech",
            Path.home() / "fish-speech",
        ]
        for p in candidates:
            if p.exists() and (p / "fish_speech").exists():
                return p
        return candidates[0]

    def _find_checkpoint(self) -> str:
        """Locate the Fish Speech checkpoint directory."""
        candidates = [
            self.config.models_dir / "fish-speech" / "fish-speech-1.5",
            self.config.models_dir / "fish-speech",
            self._get_fish_speech_dir() / "checkpoints" / "fish-speech-1.5",
        ]
        for p in candidates:
            if p.exists() and (p / "config.json").exists():
                return str(p)

        from huggingface_hub import snapshot_download
        logger.info("Downloading Fish Speech 1.5 checkpoint...")
        path = snapshot_download(
            "fishaudio/fish-speech-1.5",
            local_dir=str(self.config.models_dir / "fish-speech" / "fish-speech-1.5"),
        )
        return path

    def _load_engine(self):
        """Load Fish Speech engine — tries Python API (v2 then v1), falls back to CLI."""
        if self._engine is not None:
            return self._engine

        import torch

        checkpoint_path = self._find_checkpoint()
        self._checkpoint_path = checkpoint_path
        device = f"cuda:{self.config.primary_gpu_id}" if self.config.use_gpu else "cpu"

        # Try Fish Speech 2.0 API first
        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            logger.info(f"Loading Fish Speech 2.0 engine from {checkpoint_path} on {device}")

            self._engine = TTSInferenceEngine(
                checkpoint_path=checkpoint_path,
                device=device,
            )
            logger.info("Fish Speech 2.0 engine loaded")
            return self._engine
        except (ImportError, TypeError) as e:
            logger.info(f"Fish Speech 2.0 API not available: {e}")

        # Try Fish Speech 1.5 API
        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.models.vqgan.inference import load_model as load_decoder_model

            logger.info(f"Loading Fish Speech 1.5 engine from {checkpoint_path}")

            decoder_ckpt = None
            for name in [
                "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                "codec.pth",
            ]:
                p = Path(checkpoint_path) / name
                if p.exists():
                    decoder_ckpt = str(p)
                    break

            if not decoder_ckpt:
                raise FileNotFoundError(f"No decoder checkpoint in {checkpoint_path}")

            decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=decoder_ckpt,
                device=device,
            )
            precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            llm_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=False,
            )
            self._engine = TTSInferenceEngine(
                llm_queue=llm_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=False,
            )
            logger.info(f"Fish Speech 1.5 engine loaded on {device}")
            return self._engine

        except (ImportError, Exception) as e:
            logger.info(f"Fish Speech Python API not available: {e}")
            logger.info("Falling back to Fish Speech CLI")
            self._engine = "cli"
            return self._engine

    def _synthesize_via_cli(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
    ):
        """Fallback: use Fish Speech CLI for synthesis."""
        fish_speech_dir = self._get_fish_speech_dir()
        checkpoint_path = self._checkpoint_path or self._find_checkpoint()

        if not fish_speech_dir.exists():
            raise FileNotFoundError(
                f"Fish Speech directory not found at {fish_speech_dir}. "
                f"Run setup.sh to clone the repo."
            )

        venv_python = Path(self.config.base_dir) / "backend" / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(fish_speech_dir)

        try:
            result = subprocess.run(
                [
                    python_cmd, "-c",
                    f"""
import sys
sys.path.insert(0, '{fish_speech_dir}')
import torch
import soundfile as sf
from pathlib import Path

checkpoint_path = '{checkpoint_path}'
ref_audio = '{speaker_wav}'
text = '''{text.replace("'", "\\'")}'''
output = '{output_path}'
device = 'cuda:{self.config.primary_gpu_id}'

try:
    from fish_speech.inference_engine import TTSInferenceEngine
    engine = TTSInferenceEngine(checkpoint_path=checkpoint_path, device=device)
    
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    ref_bytes = open(ref_audio, 'rb').read()
    request = ServeTTSRequest(
        text=text,
        references=[ServeReferenceAudio(audio=ref_bytes, text='')],
        format='wav',
        streaming=False,
    )
    chunks = list(engine.inference(request))
    audio_data = b''.join(chunks)
    with open(output, 'wb') as f:
        f.write(audio_data)
    print('OK')
except Exception as e:
    print(f'FISH_ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
"""
                ],
                cwd=str(fish_speech_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Fish Speech CLI failed: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Fish Speech TTS timed out after 120s")

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
        engine = self._load_engine()

        if engine == "cli":
            self._synthesize_via_cli(text, speaker_wav, output_path)
        else:
            try:
                from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
            except ImportError:
                self._synthesize_via_cli(text, speaker_wav, output_path)
                info = sf.info(output_path)
                return info.duration

            ref_audio_bytes = open(speaker_wav, "rb").read()

            request = ServeTTSRequest(
                text=text,
                references=[
                    ServeReferenceAudio(
                        audio=ref_audio_bytes,
                        text="",
                    )
                ],
                format="wav",
                streaming=False,
            )

            result_chunks = list(engine.inference(request))
            audio_data = b"".join(result_chunks)

            with open(output_path, "wb") as f:
                f.write(audio_data)

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
        """Time-stretch audio to fit within target duration using librosa."""
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
        """Synthesize all translated segments with voice cloning."""
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
        """Release GPU memory."""
        import torch

        if self._engine is not None and self._engine != "cli":
            del self._engine
        self._engine = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TTS model unloaded.")
