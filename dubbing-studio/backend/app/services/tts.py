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
        gpu_id: int | None = None,
        worker_tag: str = "",
    ) -> list[dict]:
        """Run a batch of TTS segments in an isolated subprocess.

        Writes input JSON to a temp file, calls tts_worker.py, reads
        results from stdout. When the process exits, ALL GPU memory
        is freed by the OS.
        """
        if gpu_id is None:
            gpu_id = self.config.primary_gpu_id

        fish_speech_dir = self._get_fish_speech_dir()

        worker_input = {
            "batch": batch,
            "output_dir": output_dir,
            "gpu_id": gpu_id,
            "models_dir": str(self.config.models_dir),
            "fish_speech_dir": fish_speech_dir,
            "max_ref_seconds": 10,
            "max_text_chars": 400,
            "max_new_tokens": 600,
        }

        timeout = max(120, len(batch) * timeout_per_segment)

        tag = worker_tag or str(os.getpid())
        input_path = os.path.join(output_dir, f"_worker_input_{tag}.json")
        output_json_path = os.path.join(output_dir, f"_worker_output_{tag}.json")
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
                f"Launching TTS worker [{tag}]: {len(batch)} segments, "
                f"timeout={timeout}s, gpu={gpu_id}"
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

            if os.path.exists(output_json_path):
                with open(output_json_path) as f:
                    results = json.load(f)
                if result.returncode != 0:
                    logger.warning(
                        f"TTS worker crashed on exit (rc={result.returncode}) "
                        f"but results were saved — treating as success"
                    )
                return results

            if result.returncode != 0:
                logger.error(f"TTS worker failed (rc={result.returncode})")
                if result.stderr:
                    logger.error(f"stderr (last 3000): {result.stderr[-3000:]}")
                if result.stdout:
                    logger.error(f"stdout (last 1000): {result.stdout[-1000:]}")

                if result.returncode in (-11, -6):
                    logger.warning(
                        f"Signal {-result.returncode} detected — retrying on CPU..."
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

                    if os.path.exists(output_json_path):
                        with open(output_json_path) as f:
                            return json.load(f)

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
        batch_size: int = 1,
    ) -> list:
        """Synthesize all translated segments with voice cloning.

        Distributes batches across both GPUs in parallel. Each batch runs
        in a completely separate Python process. Safe for 2+ hour videos
        (500+ segments).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

        gpu_ids = [self.config.primary_gpu_id, self.config.secondary_gpu_id]
        num_gpus = len(gpu_ids)

        all_batches = []
        for batch_start in range(0, total_work, batch_size):
            all_batches.append(work_items[batch_start : batch_start + batch_size])

        total_batches = len(all_batches)
        logger.info(
            f"Synthesizing {total_work} segments in {total_batches} batches "
            f"(batch_size={batch_size}) across {num_gpus} GPUs: {gpu_ids}"
        )

        completed = 0
        batch_idx = 0

        while batch_idx < total_batches:
            parallel = []
            for g in range(num_gpus):
                if batch_idx + g < total_batches:
                    parallel.append((batch_idx + g, all_batches[batch_idx + g], gpu_ids[g]))

            if len(parallel) == 1:
                bi, batch, gid = parallel[0]
                batch_indices = [b["index"] for b in batch]
                logger.info(f"Batch {bi+1}/{total_batches} [GPU {gid}]: segments {batch_indices}")
                results = self._run_worker_batch(
                    batch, output_dir, gpu_id=gid,
                    worker_tag=f"g{gid}_b{bi}",
                )
                self._apply_results(results, segments)
                completed += len(batch)
            else:
                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    futures = {}
                    for bi, batch, gid in parallel:
                        batch_indices = [b["index"] for b in batch]
                        logger.info(
                            f"Batch {bi+1}/{total_batches} [GPU {gid}]: segments {batch_indices}"
                        )
                        future = executor.submit(
                            self._run_worker_batch,
                            batch, output_dir, gpu_id=gid,
                            worker_tag=f"g{gid}_b{bi}",
                        )
                        futures[future] = (bi, batch)

                    for future in as_completed(futures):
                        bi, batch = futures[future]
                        try:
                            results = future.result()
                            self._apply_results(results, segments)
                        except Exception as e:
                            logger.error(f"Batch {bi+1} raised exception: {e}")
                            for item in batch:
                                seg = segments[item["index"]]
                                seg.synth_audio_path = None
                                seg.synth_duration = 0.0
                        completed += len(batch)

            batch_idx += len(parallel)

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

    def _apply_results(self, results: list[dict], segments: list):
        """Apply worker results to segment objects."""
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

    def unload(self):
        pass
