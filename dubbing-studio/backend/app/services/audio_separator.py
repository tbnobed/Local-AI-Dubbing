"""
Audio source separation using Demucs (htdemucs).

Separates audio into vocals and instrumentals (drums + bass + other).
Runs once per video. The clean vocal stem produces much better voice
samples for Fish Speech cloning, and the instrumental stem is mixed
back under the dubbed vocals in the final output.
"""
import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioSeparatorService:
    def __init__(self, config):
        self.config = config

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        device: str = "cuda:0",
        model: str = "htdemucs",
    ) -> dict[str, str]:
        """
        Separate audio into vocals and instrumental stems using Demucs.

        Returns dict with keys:
          - "vocals": path to isolated vocal track
          - "no_vocals": path to instrumental track (drums+bass+other mixed)
          - "original": path to the original audio (fallback)

        Demucs outputs 4 stems: vocals, drums, bass, other.
        We combine drums+bass+other into a single instrumental track.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = output_dir / "vocals.wav"
        no_vocals_path = output_dir / "no_vocals.wav"

        if vocals_path.exists() and no_vocals_path.exists():
            logger.info("Demucs stems already exist, skipping separation")
            return {
                "vocals": str(vocals_path),
                "no_vocals": str(no_vocals_path),
                "original": str(audio_path),
            }

        demucs_out = output_dir / "demucs_raw"
        demucs_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running Demucs ({model}) on {device}...")

        cmd = [
            "python3", "-m", "demucs",
            "--two-stems", "vocals",
            "-n", model,
            "-d", device,
            "-o", str(demucs_out),
            "--filename", "{stem}.{ext}",
            str(audio_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"Demucs failed (rc={result.returncode})")
                logger.error(f"stderr: {result.stderr[-2000:]}")
                return self._fallback(audio_path, output_dir)

        except subprocess.TimeoutExpired:
            logger.error("Demucs timed out after 600s")
            return self._fallback(audio_path, output_dir)
        except FileNotFoundError:
            logger.error("Demucs not installed — run: pip install demucs")
            return self._fallback(audio_path, output_dir)

        stem_dir = demucs_out / model
        if not stem_dir.exists():
            for d in demucs_out.iterdir():
                if d.is_dir():
                    stem_dir = d
                    break

        demucs_vocals = stem_dir / "vocals.wav"
        demucs_no_vocals = stem_dir / "no_vocals.wav"

        if not demucs_vocals.exists():
            logger.error(f"Demucs vocals stem not found at {demucs_vocals}")
            found = list(stem_dir.glob("*.wav")) if stem_dir.exists() else []
            logger.error(f"Found stems: {[f.name for f in found]}")
            return self._fallback(audio_path, output_dir)

        shutil.move(str(demucs_vocals), str(vocals_path))

        if demucs_no_vocals.exists():
            shutil.move(str(demucs_no_vocals), str(no_vocals_path))
        else:
            self._mix_instrumentals(stem_dir, no_vocals_path)

        shutil.rmtree(str(demucs_out), ignore_errors=True)

        logger.info(f"Demucs separation complete: vocals={vocals_path.stat().st_size // 1024}KB, "
                     f"instrumentals={no_vocals_path.stat().st_size // 1024}KB")

        return {
            "vocals": str(vocals_path),
            "no_vocals": str(no_vocals_path),
            "original": str(audio_path),
        }

    def _mix_instrumentals(self, stem_dir: Path, output_path: Path):
        """Mix drums + bass + other stems into a single instrumental track."""
        import ffmpeg

        stems = []
        for name in ["drums.wav", "bass.wav", "other.wav"]:
            p = stem_dir / name
            if p.exists():
                stems.append(p)

        if not stems:
            logger.warning("No instrumental stems found")
            return

        if len(stems) == 1:
            shutil.move(str(stems[0]), str(output_path))
            return

        inputs = [ffmpeg.input(str(s)) for s in stems]
        mixed = ffmpeg.filter(
            [i.audio for i in inputs],
            "amix",
            inputs=len(stems),
            duration="longest",
            normalize=0,
        )
        ffmpeg.output(mixed, str(output_path), acodec="pcm_s16le").overwrite_output().run(quiet=True)

    def _fallback(self, audio_path: Path, output_dir: Path) -> dict[str, str]:
        """Fallback when Demucs fails — use original audio for everything."""
        logger.warning("Demucs separation failed — falling back to original audio")
        return {
            "vocals": str(audio_path),
            "no_vocals": None,
            "original": str(audio_path),
        }
