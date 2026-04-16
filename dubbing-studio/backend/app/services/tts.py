"""
Text-to-Speech with voice cloning using Fish Speech.

Each batch of segments runs in a separate Python process via tts_worker.py.
When the process exits, the OS reclaims all GPU memory — no stale threads,
no CUDA state leaks, reliable even for 2+ hour videos with 500+ segments.
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).parent / "tts_worker.py"


class TTSService:
    def __init__(self, config):
        self.config = config

    def _get_fish_speech_dir(self) -> str | None:
        candidates = [
            Path(self.config.base_dir) / "fish-speech",
            Path(self.config.base_dir).parent / "fish-speech",
            Path.home() / "fish-speech",
        ]
        for p in candidates:
            if p.exists() and (p / "fish_speech").exists():
                return str(p)
        return None

    def _get_python_cmd(self) -> str:
        venv_python = Path(self.config.base_dir) / "backend" / "venv" / "bin" / "python3"
        if venv_python.exists():
            return str(venv_python)
        return "python3"

    def _run_worker_batch(
        self,
        batch: list[dict],
        output_dir: str,
        timeout_per_segment: int = 60,
    ) -> list[dict]:
        """Run a batch of TTS segments in an isolated subprocess.

        Writes input JSON to a temp file, calls tts_worker.py, reads
        results from stdout. When the process exits, ALL GPU memory
        is freed by the OS.
        """
        fish_speech_dir = self._get_fish_speech_dir()

        worker_input = {
            "batch": batch,
            "output_dir": output_dir,
            "gpu_id": self.config.primary_gpu_id,
            "models_dir": str(self.config.models_dir),
            "fish_speech_dir": fish_speech_dir,
            "max_ref_seconds": 10,
            "max_text_chars": 300,
            "max_new_tokens": 300,
        }

        timeout = max(120, len(batch) * timeout_per_segment)

        input_path = os.path.join(output_dir, f"_worker_input_{os.getpid()}.json")
        output_json_path = os.path.join(output_dir, f"_worker_output_{os.getpid()}.json")
        worker_input["output_json_path"] = output_json_path

        with open(input_path, "w") as f:
            json.dump(worker_input, f)

        try:
            python_cmd = self._get_python_cmd()

            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/root"),
                "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            }
            for k in list(env):
                if not env[k]:
                    del env[k]

            if fish_speech_dir:
                env["PYTHONPATH"] = fish_speech_dir

            logger.info(
                f"Launching TTS worker: {len(batch)} segments, "
                f"timeout={timeout}s, gpu={self.config.primary_gpu_id}"
            )
            logger.info(f"Worker script: {WORKER_SCRIPT}")
            logger.info(f"Python: {python_cmd}")
            logger.info(f"PYTHONPATH: {env.get('PYTHONPATH', '(not set)')}")

            result = subprocess.run(
                [python_cmd, str(WORKER_SCRIPT), input_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(Path(self.config.base_dir) / "backend"),
                start_new_session=True,
            )

            if result.returncode != 0:
                logger.error(f"TTS worker failed (rc={result.returncode})")
                if result.stderr:
                    logger.error(f"stderr (last 3000): {result.stderr[-3000:]}")
                if result.stdout:
                    logger.error(f"stdout (last 1000): {result.stdout[-1000:]}")

                if result.returncode == -11:
                    logger.warning(
                        "SIGSEGV detected — retrying batch on CPU..."
                    )
                    worker_input["gpu_id"] = -1
                    with open(input_path, "w") as f:
                        json.dump(worker_input, f)

                    result = subprocess.run(
                        [python_cmd, str(WORKER_SCRIPT), input_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout * 3,
                        env=env,
                        cwd=str(Path(self.config.base_dir) / "backend"),
                        start_new_session=True,
                    )

                    if result.returncode != 0:
                        logger.error(f"TTS worker CPU fallback also failed (rc={result.returncode})")
                        if result.stderr:
                            logger.error(f"stderr: {result.stderr[-3000:]}")
                        return [
                            {"index": item["index"], "success": False,
                             "error": f"Worker exit code {result.returncode} (GPU+CPU)"}
                            for item in batch
                        ]
                else:
                    return [
                        {"index": item["index"], "success": False,
                         "error": f"Worker exit code {result.returncode}"}
                        for item in batch
                    ]

            if os.path.exists(output_json_path):
                with open(output_json_path) as f:
                    results = json.load(f)
                return results

            try:
                results = json.loads(result.stdout)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"No output JSON file and stdout parse failed: {e}")
                logger.error(f"stdout (last 500): {result.stdout[-500:]}")
                return [
                    {"index": item["index"], "success": False,
                     "error": "Invalid worker output"}
                    for item in batch
                ]

            return results

        except subprocess.TimeoutExpired:
            logger.error(f"TTS worker timed out after {timeout}s — killing")
            return [
                {"index": item["index"], "success": False, "error": "timeout"}
                for item in batch
            ]
        finally:
            for p in [input_path, output_json_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

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

        Processes segments in batches. Each batch runs in a completely
        separate Python process. Safe for 2+ hour videos (500+ segments).
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        default_speaker = (
            list(speaker_samples.values())[0] if speaker_samples else None
        )

        work_items = []
        for i, seg in enumerate(segments):
            translated_text = getattr(seg, "translated_text", seg.text)
            if not translated_text.strip():
                seg.synth_audio_path = None
                seg.synth_duration = 0.0
                continue

            speaker_id = getattr(seg, "speaker", "SPEAKER_00")
            speaker_wav = speaker_samples.get(speaker_id, default_speaker)

            if not speaker_wav:
                logger.warning(f"No voice sample for {speaker_id}, skipping segment {i}")
                seg.synth_audio_path = None
                seg.synth_duration = 0.0
                continue

            work_items.append({
                "index": i,
                "text": translated_text,
                "speaker_wav": speaker_wav,
                "original_duration": seg.end - seg.start,
            })

        total_work = len(work_items)
        if total_work == 0:
            logger.warning("No segments to synthesize")
            return segments

        logger.info(
            f"Synthesizing {total_work} segments in batches of {batch_size} "
            f"(~{(total_work + batch_size - 1) // batch_size} subprocess calls)"
        )

        completed = 0
        for batch_start in range(0, total_work, batch_size):
            batch = work_items[batch_start : batch_start + batch_size]
            batch_indices = [b["index"] for b in batch]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_work + batch_size - 1) // batch_size
            logger.info(
                f"Batch {batch_num}/{total_batches}: segments {batch_indices}"
            )

            results = self._run_worker_batch(batch, output_dir)

            for result in results:
                idx = result["index"]
                seg = segments[idx]
                if result.get("success"):
                    seg.synth_audio_path = result["stretched_path"]
                    seg.synth_duration = result.get("synth_duration", 0.0)
                    logger.info(
                        f"Segment {idx}: {result.get('synth_duration', 0):.1f}s "
                        f"-> {result.get('original_duration', 0):.1f}s slot"
                    )
                else:
                    seg.synth_audio_path = None
                    seg.synth_duration = 0.0
                    logger.error(
                        f"Segment {idx} failed: {result.get('error', 'unknown')}"
                    )

            completed += len(batch)
            if progress_callback:
                progress_callback(completed / total_work)

        for seg in segments:
            if not hasattr(seg, "synth_audio_path"):
                seg.synth_audio_path = None
                seg.synth_duration = 0.0

        success_count = sum(
            1 for seg in segments
            if getattr(seg, "synth_audio_path", None) is not None
        )
        logger.info(f"TTS complete: {success_count}/{len(segments)} segments synthesized")
        return segments

    def unload(self):
        pass
