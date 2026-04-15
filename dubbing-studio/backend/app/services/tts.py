"""
Text-to-Speech with voice cloning using Fish Speech 1.5.

Fish Speech is installed from source (github.com/fishaudio/fish-speech)
and provides the TTSInferenceEngine for programmatic usage.

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
        self._decoder_model = None
        self._llm_queue = None
        self._codec = None
        self._sample_rate = 44100
        self._checkpoint_path = None

    def _find_checkpoint(self) -> str:
        """Locate the Fish Speech checkpoint directory."""
        candidates = [
            self.config.models_dir / "fish-speech" / "fish-speech-1.5",
            self.config.models_dir / "fish-speech",
            Path(self.config.base_dir).parent / "fish-speech" / "checkpoints" / "fish-speech-1.5",
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
        """Load the Fish Speech TTSInferenceEngine."""
        if self._engine is not None:
            return self._engine

        import torch

        checkpoint_path = self._find_checkpoint()
        self._checkpoint_path = checkpoint_path
        device = f"cuda:{self.config.primary_gpu_id}" if self.config.use_gpu else "cpu"

        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.models.vqgan.inference import load_model as load_decoder_model

            logger.info(f"Loading Fish Speech decoder from {checkpoint_path}")

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
                raise FileNotFoundError(
                    f"No decoder checkpoint found in {checkpoint_path}. "
                    f"Expected firefly-gan-vq-fsq-8x1024-21hz-generator.pth or codec.pth"
                )

            self._decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=decoder_ckpt,
                device=device,
            )

            precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            self._llm_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=False,
            )

            self._engine = TTSInferenceEngine(
                llm_queue=self._llm_queue,
                decoder_model=self._decoder_model,
                precision=precision,
                compile=False,
            )

            logger.info(f"Fish Speech TTSInferenceEngine loaded on {device}")
            return self._engine

        except ImportError:
            logger.info("TTSInferenceEngine not available, falling back to CLI pipeline")
            self._engine = "cli"
            return self._engine

    def _synthesize_via_cli(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
    ):
        """Fallback: use Fish Speech CLI tools for synthesis."""
        fish_speech_dir = Path(self.config.base_dir).parent / "fish-speech"
        checkpoint_path = self._checkpoint_path or self._find_checkpoint()
        temp_dir = Path(output_path).parent

        decoder_ckpt = None
        for name in [
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "codec.pth",
        ]:
            p = Path(checkpoint_path) / name
            if p.exists():
                decoder_ckpt = str(p)
                break

        fake_npy = str(temp_dir / "ref_tokens.npy")
        codes_npy = str(temp_dir / "codes_0.npy")

        # Step 1: Encode reference audio
        cmd1 = [
            "python3", str(fish_speech_dir / "fish_speech" / "models" / "dac" / "inference.py"),
            "-i", speaker_wav,
            "--checkpoint-path", decoder_ckpt or checkpoint_path,
        ]
        logger.info(f"Fish Speech CLI step 1: encoding reference audio")
        subprocess.run(cmd1, check=True, cwd=str(fish_speech_dir), capture_output=True)

        # Step 2: Generate semantic tokens
        cmd2 = [
            "python3", str(fish_speech_dir / "fish_speech" / "models" / "text2semantic" / "inference.py"),
            "--text", text,
            "--prompt-text", "",
            "--prompt-tokens", fake_npy,
            "--checkpoint-path", checkpoint_path,
        ]
        logger.info(f"Fish Speech CLI step 2: generating semantic tokens")
        subprocess.run(cmd2, check=True, cwd=str(fish_speech_dir), capture_output=True)

        # Step 3: Decode to audio
        cmd3 = [
            "python3", str(fish_speech_dir / "fish_speech" / "models" / "dac" / "inference.py"),
            "-i", codes_npy,
        ]
        logger.info(f"Fish Speech CLI step 3: decoding to audio")
        subprocess.run(cmd3, check=True, cwd=str(fish_speech_dir), capture_output=True)

        # Move output
        cli_output = Path(fish_speech_dir) / "fake.wav"
        if cli_output.exists():
            import shutil
            shutil.move(str(cli_output), output_path)
        else:
            raise FileNotFoundError("Fish Speech CLI did not produce output audio")

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
            from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

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
            del self._decoder_model
            del self._llm_queue
        self._engine = None
        self._decoder_model = None
        self._llm_queue = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TTS model unloaded.")
