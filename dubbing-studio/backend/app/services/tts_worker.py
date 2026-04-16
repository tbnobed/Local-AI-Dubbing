#!/usr/bin/env python3
"""
Standalone TTS worker script — runs in its own process.

Called by tts.py via subprocess.run(). Loads Fish Speech, processes a batch
of segments, writes WAV files to disk, outputs results as JSON to stdout.

When this script exits, the OS reclaims ALL GPU memory automatically.
No stale threads, no CUDA state leaks between batches.
"""
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TTS-worker] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("tts_worker")


def find_checkpoint(models_dir: Path) -> str:
    for p in [
        models_dir / "fish-speech" / "fish-speech-1.5",
        models_dir / "fish-speech",
    ]:
        if p.exists() and (p / "config.json").exists():
            return str(p)
    raise FileNotFoundError(f"No Fish Speech checkpoint found under {models_dir}")


def find_decoder_ckpt(checkpoint_path: str) -> str:
    for name in [
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        "codec.pth",
    ]:
        p = Path(checkpoint_path) / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No decoder checkpoint in {checkpoint_path}")


def time_stretch(raw_path: str, stretched_path: str, original_duration: float):
    import soundfile as sf

    info = sf.info(raw_path)
    current_dur = info.duration
    if current_dur <= 0:
        shutil.copy2(raw_path, stretched_path)
        return

    ratio = current_dur / original_duration
    if abs(ratio - 1.0) < 0.05:
        shutil.copy2(raw_path, stretched_path)
        return

    if ratio > 1.5:
        log.warning(
            f"Synth audio {current_dur:.1f}s is much longer than slot {original_duration:.1f}s "
            f"({ratio:.1f}x) — truncating to fit"
        )
        data, sr = sf.read(raw_path)
        max_samples = int(original_duration * sr)
        if len(data) > max_samples:
            data = data[:max_samples]
        sf.write(stretched_path, data, sr)
        return

    ratio = max(0.7, min(1.5, ratio))
    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r *= 2.0
    filters.append(f"atempo={r:.4f}")

    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-filter:a", ",".join(filters),
         "-loglevel", "error", stretched_path],
        check=True, timeout=30,
    )


def synthesize_batch(batch, engine, output_dir, max_ref_seconds, max_text_chars, max_new_tokens, device="cpu"):
    import numpy as np
    import soundfile as sf
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    results = []
    for item in batch:
        idx = item["index"]
        text = item["text"]
        speaker_wav = item["speaker_wav"]
        original_duration = item["original_duration"]
        raw_path = str(Path(output_dir) / f"seg_{idx:04d}_raw.wav")
        stretched_path = str(Path(output_dir) / f"seg_{idx:04d}.wav")

        try:
            if len(text) > max_text_chars:
                log.warning(f"Segment {idx}: truncating {len(text)} chars -> {max_text_chars}")
                text = text[:max_text_chars]

            data, sr = sf.read(speaker_wav)
            max_samples = int(max_ref_seconds * sr)
            if len(data) > max_samples:
                data = data[:max_samples]
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="wav")

            tokens_for_duration = int(original_duration * 21 * 1.3)
            segment_max_tokens = max(64, min(max_new_tokens, tokens_for_duration))
            log.info(
                f"Segment {idx}: text={len(text)} chars, "
                f"slot={original_duration:.1f}s, max_tokens={segment_max_tokens}"
            )

            request = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=buf.getvalue(), text="")],
                format="wav",
                streaming=False,
                max_new_tokens=segment_max_tokens,
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
                if isinstance(audio, np.ndarray):
                    if audio.ndim > 1:
                        audio = audio.squeeze()
                    if audio.size > 0:
                        all_audio.append(audio)

            if not all_audio:
                raise RuntimeError("No audio output from Fish Speech")

            combined = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
            sf.write(raw_path, combined, sample_rate)
            synth_duration = len(combined) / sample_rate

            time_stretch(raw_path, stretched_path, original_duration)

            results.append({
                "index": idx,
                "success": True,
                "stretched_path": stretched_path,
                "synth_duration": round(synth_duration, 2),
                "original_duration": round(original_duration, 2),
            })
            log.info(f"Segment {idx}: {synth_duration:.1f}s synth -> {original_duration:.1f}s slot")

        except Exception as e:
            log.error(f"Segment {idx} failed: {e}")
            results.append({"index": idx, "success": False, "error": str(e)})

        if device != "cpu":
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: tts_worker.py <input_json_path>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path) as f:
        args = json.load(f)

    batch = args["batch"]
    output_dir = args["output_dir"]
    gpu_id = args.get("gpu_id", 0)
    models_dir = Path(args["models_dir"])
    max_ref_seconds = args.get("max_ref_seconds", 10)
    max_text_chars = args.get("max_text_chars", 300)
    max_new_tokens = args.get("max_new_tokens", 300)

    fish_speech_dir = args.get("fish_speech_dir")
    if fish_speech_dir and fish_speech_dir not in sys.path:
        sys.path.insert(0, fish_speech_dir)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    results = []
    try:
        import torch

        if gpu_id < 0 or not torch.cuda.is_available():
            device = "cpu"
            log.info("Running Fish Speech on CPU")
        else:
            device = f"cuda:{gpu_id}"

        checkpoint_path = find_checkpoint(models_dir)
        decoder_ckpt = find_decoder_ckpt(checkpoint_path)
        log.info(f"Loading Fish Speech on {device}, checkpoint={checkpoint_path}")

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.vqgan.inference import load_model as load_decoder_model

        if device == "cpu":
            precision = torch.float32
        else:
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
        log.info(f"Engine ready on {device}, processing {len(batch)} segments")

        results = synthesize_batch(
            batch, engine, output_dir,
            max_ref_seconds, max_text_chars, max_new_tokens,
            device=device,
        )

    except Exception as e:
        log.error(f"Worker failed: {e}\n{traceback.format_exc()}")
        for item in batch:
            if not any(r["index"] == item["index"] for r in results):
                results.append({"index": item["index"], "success": False, "error": str(e)})

    output_json_path = args.get("output_json_path")
    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(results, f)
        log.info(f"Results written to {output_json_path}")
    else:
        json.dump(results, sys.stdout)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
