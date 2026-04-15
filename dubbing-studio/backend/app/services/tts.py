"""
Text-to-Speech with voice cloning using Fish Speech.

Supports Fish Speech 2.0 (OpenAudio S1-mini) with DAC decoder,
and Fish Speech 1.5 with VQGAN decoder.

Pipeline:
  1. Encode reference audio → voice tokens
  2. Generate semantic tokens from text + voice prompt
  3. Decode tokens → waveform
"""
import logging
import os
import subprocess
import tempfile
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
        """Locate or download the Fish Speech 1.5 checkpoint directory."""
        candidates = [
            self.config.models_dir / "fish-speech" / "fish-speech-1.5",
            self.config.models_dir / "fish-speech",
            self._get_fish_speech_dir() / "checkpoints" / "fish-speech-1.5",
        ]
        for p in candidates:
            if p.exists() and (p / "config.json").exists():
                logger.info(f"Found Fish Speech 1.5 checkpoint at {p}")
                return str(p)

        from huggingface_hub import snapshot_download
        logger.info("Downloading Fish Speech 1.5 checkpoint...")
        path = snapshot_download(
            "fishaudio/fish-speech-1.5",
            local_dir=str(self.config.models_dir / "fish-speech" / "fish-speech-1.5"),
        )
        return path

    def _load_engine(self):
        """Load Fish Speech 1.5 engine — tries Python API, falls back to CLI."""
        if self._engine is not None:
            return self._engine

        import torch

        checkpoint_path = self._find_checkpoint()
        self._checkpoint_path = checkpoint_path
        device = f"cuda:{self.config.primary_gpu_id}" if self.config.use_gpu else "cpu"
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Try Fish Speech 1.5 Python API (VQGAN decoder)
        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.models.vqgan.inference import load_model as load_decoder_model

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

            logger.info(f"Loading Fish Speech 1.5 decoder from {decoder_ckpt}")
            decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=decoder_ckpt,
                device=device,
            )

            logger.info("Loading Fish Speech 1.5 LLM...")
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=False,
            )

            self._engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=False,
            )
            logger.info(f"Fish Speech 1.5 engine loaded on {device}")
            return self._engine

        except (ImportError, Exception) as e:
            logger.info(f"Fish Speech Python API not available: {e}")

        # Fall back to CLI-based 3-step pipeline
        logger.info("Falling back to Fish Speech CLI pipeline")
        self._engine = "cli"
        return self._engine

    def _synthesize_via_cli(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
    ):
        """Fallback: 3-step CLI pipeline using fish_speech scripts."""
        fish_speech_dir = self._get_fish_speech_dir()
        checkpoint_path = self._checkpoint_path or self._find_checkpoint()

        venv_python = Path(self.config.base_dir) / "backend" / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(fish_speech_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_npy = os.path.join(tmpdir, "fake.npy")
            codes_npy = os.path.join(tmpdir, "codes_0.npy")

            # Determine which encoder/decoder script exists
            dac_script = fish_speech_dir / "fish_speech" / "models" / "dac" / "inference.py"
            vqgan_script = fish_speech_dir / "fish_speech" / "models" / "vqgan" / "inference.py"

            if dac_script.exists():
                codec_script = str(dac_script)
            elif vqgan_script.exists():
                codec_script = str(vqgan_script)
            else:
                raise FileNotFoundError(
                    f"No codec inference script found in {fish_speech_dir}. "
                    f"Expected dac/inference.py or vqgan/inference.py"
                )

            text2sem_script = str(
                fish_speech_dir / "fish_speech" / "models" / "text2semantic" / "inference.py"
            )
            if not Path(text2sem_script).exists():
                raise FileNotFoundError(f"text2semantic inference script not found: {text2sem_script}")

            # Step 1: Encode reference audio → voice tokens
            cmd1 = [
                python_cmd, codec_script,
                "-i", speaker_wav,
                "--checkpoint-path", os.path.join(checkpoint_path, "codec.pth"),
                "--output-path", tmpdir,
            ]
            # Also try with firefly checkpoint name
            codec_ckpt = os.path.join(checkpoint_path, "codec.pth")
            if not os.path.exists(codec_ckpt):
                codec_ckpt = os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
            cmd1 = [
                python_cmd, codec_script,
                "-i", speaker_wav,
                "--checkpoint-path", codec_ckpt,
                "--output-path", tmpdir,
            ]

            r1 = subprocess.run(cmd1, cwd=str(fish_speech_dir), env=env,
                                capture_output=True, text=True, timeout=60)
            if r1.returncode != 0:
                raise RuntimeError(f"Codec encode failed: {r1.stderr[:500]}")

            # Find the generated npy file
            npy_files = list(Path(tmpdir).glob("*.npy"))
            if not npy_files:
                raise RuntimeError("Codec encode produced no .npy files")
            fake_npy = str(npy_files[0])

            # Step 2: Generate semantic tokens from text + voice prompt
            safe_text = text.replace("'", "\\'").replace('"', '\\"')
            cmd2 = [
                python_cmd, text2sem_script,
                "--text", text,
                "--prompt-tokens", fake_npy,
                "--checkpoint-path", checkpoint_path,
                "--output-path", tmpdir,
                "--num-samples", "1",
            ]
            r2 = subprocess.run(cmd2, cwd=str(fish_speech_dir), env=env,
                                capture_output=True, text=True, timeout=120)
            if r2.returncode != 0:
                raise RuntimeError(f"Text2semantic failed: {r2.stderr[:500]}")

            # Find generated codes file
            code_files = sorted(Path(tmpdir).glob("codes_*.npy"))
            if not code_files:
                raise RuntimeError("Text2semantic produced no codes files")

            # Step 3: Decode semantic tokens → waveform
            cmd3 = [
                python_cmd, codec_script,
                "-i", str(code_files[0]),
                "--checkpoint-path", codec_ckpt,
                "--output-path", tmpdir,
            ]
            r3 = subprocess.run(cmd3, cwd=str(fish_speech_dir), env=env,
                                capture_output=True, text=True, timeout=60)
            if r3.returncode != 0:
                raise RuntimeError(f"Codec decode failed: {r3.stderr[:500]}")

            # Find output wav
            wav_files = list(Path(tmpdir).glob("*.wav"))
            if not wav_files:
                raise RuntimeError("Codec decode produced no .wav files")

            import shutil
            shutil.copy2(str(wav_files[0]), output_path)

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

            all_audio = []
            sample_rate = 44100

            for result in engine.inference(request):
                if hasattr(result, "audio") and hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate
                    audio = result.audio
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    if isinstance(audio, np.ndarray):
                        if audio.ndim > 1:
                            audio = audio.squeeze()
                        all_audio.append(audio)
                    elif isinstance(audio, bytes):
                        with open(output_path, "wb") as f:
                            f.write(audio)
                        info = sf.info(output_path)
                        return info.duration
                elif isinstance(result, bytes):
                    with open(output_path, "wb") as f:
                        f.write(result)
                    info = sf.info(output_path)
                    return info.duration

            if all_audio:
                combined = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
                sf.write(output_path, combined, sample_rate)
            else:
                raise RuntimeError("Fish Speech inference produced no audio output")

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
