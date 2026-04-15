"""
Text-to-Speech with voice cloning using Fish Speech.

Supports Fish Speech 2.0 (OpenAudio S1-mini) with DAC decoder,
and Fish Speech 1.5 with VQGAN decoder.

Pipeline:
  1. Encode reference audio → voice tokens
  2. Generate semantic tokens from text + voice prompt
  3. Decode tokens → waveform
"""
import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _tts_subprocess_worker(args: dict) -> list:
    """Run a batch of TTS segments in an isolated subprocess.
    This function runs in a spawned process with its own CUDA context.
    When it returns, the process exits and ALL GPU memory is freed by the OS.
    """
    import os
    import sys
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _log = _logging.getLogger("tts_subprocess")

    batch = args["batch"]
    output_dir = args["output_dir"]
    config_dict = args["config_dict"]

    fish_speech_dir = config_dict.get("fish_speech_dir")
    if fish_speech_dir and fish_speech_dir not in sys.path:
        sys.path.insert(0, fish_speech_dir)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    max_ref_seconds = args.get("max_ref_seconds", 10)
    max_text_chars = args.get("max_text_chars", 300)
    max_new_tokens = args.get("max_new_tokens", 300)

    results = []

    try:
        import torch
        import soundfile as _sf
        import numpy as _np
        import io
        import subprocess as _sp
        import shutil

        models_dir = Path(config_dict["models_dir"])
        gpu_id = config_dict["primary_gpu_id"]
        device = f"cuda:{gpu_id}"

        checkpoint_candidates = [
            models_dir / "fish-speech" / "fish-speech-1.5",
            models_dir / "fish-speech",
        ]
        checkpoint_path = None
        for p in checkpoint_candidates:
            if p.exists() and (p / "config.json").exists():
                checkpoint_path = str(p)
                break
        if not checkpoint_path:
            raise FileNotFoundError("Fish Speech checkpoint not found")

        _log.info(f"Subprocess loading Fish Speech on {device}, checkpoint={checkpoint_path}")

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.vqgan.inference import load_model as load_decoder_model
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        decoder_ckpt = None
        for name in ["firefly-gan-vq-fsq-8x1024-21hz-generator.pth", "codec.pth"]:
            p = Path(checkpoint_path) / name
            if p.exists():
                decoder_ckpt = str(p)
                break
        if not decoder_ckpt:
            raise FileNotFoundError(f"No decoder checkpoint in {checkpoint_path}")

        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=decoder_ckpt,
            device=device,
        )

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            compile=False,
        )

        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )

        _log.info(f"Fish Speech engine ready, processing {len(batch)} segments")

        for item in batch:
            idx = item["index"]
            text = item["text"]
            speaker_wav = item["speaker_wav"]
            original_duration = item["original_duration"]

            raw_path = str(Path(output_dir) / f"seg_{idx:04d}_raw.wav")
            stretched_path = str(Path(output_dir) / f"seg_{idx:04d}.wav")

            try:
                if len(text) > max_text_chars:
                    _log.warning(f"Truncating text ({len(text)} chars) to {max_text_chars}")
                    text = text[:max_text_chars]

                data, sr = _sf.read(speaker_wav)
                max_samples = int(max_ref_seconds * sr)
                if len(data) > max_samples:
                    data = data[:max_samples]
                    _log.info(f"Trimmed reference audio to {max_ref_seconds}s")
                buf = io.BytesIO()
                _sf.write(buf, data, sr, format="wav")
                ref_bytes = buf.getvalue()

                request = ServeTTSRequest(
                    text=text,
                    references=[ServeReferenceAudio(audio=ref_bytes, text="")],
                    format="wav",
                    streaming=False,
                    max_new_tokens=max_new_tokens,
                )

                all_audio = []
                sample_rate = 44100
                for result in engine.inference(request):
                    if hasattr(result, "error") and result.error:
                        continue
                    if hasattr(result, "sample_rate") and result.sample_rate:
                        sample_rate = result.sample_rate
                    audio = getattr(result, "audio", None)
                    if audio is None:
                        continue
                    if isinstance(audio, tuple) and len(audio) == 2:
                        sample_rate, audio = audio
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    if isinstance(audio, _np.ndarray):
                        if audio.ndim > 1:
                            audio = audio.squeeze()
                        if audio.size > 0:
                            all_audio.append(audio)

                if not all_audio:
                    raise RuntimeError("No audio output")

                combined = _np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
                _sf.write(raw_path, combined, sample_rate)
                synth_duration = len(combined) / sample_rate

                info = _sf.info(raw_path)
                current_dur = info.duration
                if current_dur > 0 and abs(current_dur / original_duration - 1.0) > 0.02:
                    rate = current_dur / original_duration
                    rate = max(0.5, min(2.0, rate))
                    atempo_filters = []
                    r = rate
                    while r > 2.0:
                        atempo_filters.append("atempo=2.0")
                        r /= 2.0
                    while r < 0.5:
                        atempo_filters.append("atempo=0.5")
                        r *= 2.0
                    atempo_filters.append(f"atempo={r:.4f}")
                    filter_str = ",".join(atempo_filters)
                    _sp.run([
                        "ffmpeg", "-y", "-i", raw_path,
                        "-filter:a", filter_str, "-loglevel", "error", stretched_path,
                    ], check=True, timeout=30)
                else:
                    shutil.copy2(raw_path, stretched_path)

                results.append({
                    "index": idx,
                    "success": True,
                    "stretched_path": stretched_path,
                    "synth_duration": synth_duration,
                    "original_duration": original_duration,
                })
                _log.info(f"Segment {idx} done: {synth_duration:.1f}s synth -> {original_duration:.1f}s slot")

            except Exception as e:
                _log.error(f"Segment {idx} failed: {e}")
                results.append({"index": idx, "success": False, "error": str(e)})

        del engine, llama_queue, decoder_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        _log.error(f"Subprocess init failed: {e}\n{traceback.format_exc()}")
        for item in batch:
            if not any(r["index"] == item["index"] for r in results):
                results.append({"index": item["index"], "success": False, "error": str(e)})

    return results


def _run_tts_batch_subprocess(
    config,
    batch: list,
    output_dir: str,
    target_language: str,
    max_ref_seconds: float = 10,
    max_text_chars: int = 300,
    max_new_tokens: int = 300,
    timeout: int = 300,
) -> list:
    """Launch a spawned subprocess to process a batch of TTS segments.
    The subprocess gets its own CUDA context; when it exits the OS
    reclaims all GPU memory. No stale threads, no CUDA state leaks.
    """
    fish_speech_dir = None
    for candidate in [
        Path(config.base_dir) / "fish-speech",
        Path(config.base_dir).parent / "fish-speech",
    ]:
        if candidate.exists() and (candidate / "fish_speech").exists():
            fish_speech_dir = str(candidate)
            break

    config_dict = {
        "base_dir": str(config.base_dir),
        "models_dir": str(config.models_dir),
        "primary_gpu_id": config.primary_gpu_id,
        "use_gpu": config.use_gpu,
        "fish_speech_dir": fish_speech_dir,
    }

    worker_args = {
        "batch": batch,
        "output_dir": output_dir,
        "target_language": target_language,
        "config_dict": config_dict,
        "max_ref_seconds": max_ref_seconds,
        "max_text_chars": max_text_chars,
        "max_new_tokens": max_new_tokens,
    }

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        async_result = pool.apply_async(_tts_subprocess_worker, (worker_args,))
        try:
            results = async_result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            logger.error(f"TTS subprocess timed out after {timeout}s, terminating")
            pool.terminate()
            results = [{"index": item["index"], "success": False, "error": "timeout"} for item in batch]

    return results


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

    def _trim_reference_audio(self, audio_path: str, max_seconds: float = 10) -> bytes:
        """Trim reference audio to max_seconds and return as bytes."""
        import io
        data, sr = sf.read(audio_path)
        max_samples = int(max_seconds * sr)
        if len(data) > max_samples:
            data = data[:max_samples]
            logger.info(f"Trimmed reference audio to {max_seconds}s ({max_samples} samples)")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="wav")
        return buf.getvalue()

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

            logger.info(f"Loading Fish Speech 1.5 decoder from {decoder_ckpt} on {device}")
            decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=decoder_ckpt,
                device=device,
            )

            logger.info(f"Loading Fish Speech 1.5 LLM on {device}...")
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

        if len(text) > 300:
            logger.warning(f"Truncating long text ({len(text)} chars) to 300 chars for TTS")
            text = text[:300]

        if engine == "cli":
            self._synthesize_via_cli(text, speaker_wav, output_path)
        else:
            try:
                from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
            except ImportError:
                self._synthesize_via_cli(text, speaker_wav, output_path)
                info = sf.info(output_path)
                return info.duration

            ref_audio_bytes = self._trim_reference_audio(speaker_wav, max_seconds=10)

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
                max_new_tokens=300,
            )

            import torch as _torch
            all_audio = []
            all_bytes = b""
            sample_rate = 44100

            for result in engine.inference(request):
                if hasattr(result, "error") and result.error:
                    logger.warning(f"Inference error: {result.error}")
                    continue

                if hasattr(result, "sample_rate") and result.sample_rate:
                    sample_rate = result.sample_rate

                audio = getattr(result, "audio", None)

                if audio is None:
                    continue

                if isinstance(audio, tuple) and len(audio) == 2:
                    sample_rate, audio = audio

                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                if isinstance(audio, np.ndarray):
                    if audio.ndim > 1:
                        audio = audio.squeeze()
                    if audio.size > 0:
                        all_audio.append(audio)
                        logger.info(f"Got audio chunk: shape={audio.shape}, sr={sample_rate}")
                elif isinstance(audio, (bytes, bytearray)) and len(audio) > 0:
                    all_bytes += bytes(audio)
                elif hasattr(audio, "read"):
                    chunk = audio.read()
                    if chunk:
                        all_bytes += chunk

            if all_audio:
                combined = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
                sf.write(output_path, combined, sample_rate)
            elif all_bytes:
                with open(output_path, "wb") as f:
                    f.write(all_bytes)
            else:
                raise RuntimeError("Fish Speech inference produced no audio output")

        info = sf.info(output_path)
        return info.duration

    def time_stretch_audio(
        self,
        audio_path: str,
        output_path: str,
        target_duration: float,
        min_rate: float = 0.5,
        max_rate: float = 2.0,
    ) -> str:
        """Time-stretch audio to fit within target duration using ffmpeg atempo."""
        import subprocess, shutil

        info = sf.info(audio_path)
        current_duration = info.duration

        if current_duration == 0:
            shutil.copy2(audio_path, output_path)
            return output_path

        rate = current_duration / target_duration
        rate = max(min_rate, min(max_rate, rate))

        if abs(rate - 1.0) < 0.02:
            shutil.copy2(audio_path, output_path)
            return output_path

        logger.info(f"Time-stretching {current_duration:.1f}s -> {target_duration:.1f}s (rate={rate:.2f})")

        atempo_filters = []
        r = rate
        while r > 2.0:
            atempo_filters.append("atempo=2.0")
            r /= 2.0
        while r < 0.5:
            atempo_filters.append("atempo=0.5")
            r *= 2.0
        atempo_filters.append(f"atempo={r:.4f}")

        filter_str = ",".join(atempo_filters)
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-filter:a", filter_str,
            "-loglevel", "error",
            output_path,
        ]
        subprocess.run(cmd, check=True, timeout=30)
        return output_path

    def synthesize_all_segments(
        self,
        segments: list,
        speaker_samples: dict[str, str],
        target_language: str,
        output_dir: str,
        progress_callback=None,
        batch_size: int = 3,
    ) -> list:
        """Synthesize all translated segments with voice cloning.
        Runs each batch in a subprocess to prevent GPU hangs on Blackwell GPUs.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        default_speaker = list(speaker_samples.values())[0] if speaker_samples else None

        batch_items = []
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

            original_duration = seg.end - seg.start
            batch_items.append({
                "index": i,
                "text": translated_text,
                "speaker_wav": speaker_wav,
                "original_duration": original_duration,
            })

        total = len(segments)

        for batch_start in range(0, len(batch_items), batch_size):
            batch = batch_items[batch_start:batch_start + batch_size]
            batch_indices = [b["index"] for b in batch]
            logger.info(f"Processing TTS batch: segments {batch_indices} ({batch_start+1}-{min(batch_start+batch_size, len(batch_items))}/{len(batch_items)})")

            results = _run_tts_batch_subprocess(
                config=self.config,
                batch=batch,
                output_dir=output_dir,
                target_language=target_language,
                max_ref_seconds=10,
                max_text_chars=300,
                max_new_tokens=300,
            )

            for result in results:
                idx = result["index"]
                seg = segments[idx]
                if result.get("success"):
                    seg.synth_audio_path = result["stretched_path"]
                    seg.synth_duration = result.get("synth_duration", 0.0)
                    logger.info(f"Segment {idx} done: {result.get('synth_duration', 0):.1f}s audio -> {result.get('original_duration', 0):.1f}s slot")
                else:
                    seg.synth_audio_path = None
                    seg.synth_duration = 0.0
                    logger.error(f"Segment {idx} failed: {result.get('error', 'unknown')}")

            if progress_callback:
                done_count = batch_start + len(batch)
                progress_callback(done_count / len(batch_items) if batch_items else 1.0)

        for seg in segments:
            if not hasattr(seg, "synth_audio_path"):
                seg.synth_audio_path = None
                seg.synth_duration = 0.0

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
