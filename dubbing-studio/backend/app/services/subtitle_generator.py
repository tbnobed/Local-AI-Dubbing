"""
SRT subtitle file generator from transcription/translation segments.
"""
import logging
from pathlib import Path
from datetime import timedelta

logger = logging.getLogger(__name__)


def seconds_to_srt_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - int(seconds)) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list, output_path: str, use_translated: bool = False) -> str:
    lines = []

    valid_segments = [s for s in segments if s.text.strip()]

    for i, seg in enumerate(valid_segments, start=1):
        text = getattr(seg, "translated_text", seg.text) if use_translated else seg.text
        text = text.strip()

        if not text:
            continue

        start_ts = seconds_to_srt_timestamp(seg.start)
        end_ts = seconds_to_srt_timestamp(seg.end)

        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    content = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"SRT written: {output_path} ({len(valid_segments)} entries)")
    return output_path


def generate_both_srts(
    segments: list,
    output_dir: str,
    job_id: str,
    source_lang: str,
    target_lang: str,
) -> tuple[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_srt = str(output_dir / f"{job_id}_original_{source_lang}.srt")
    translated_srt = str(output_dir / f"{job_id}_translated_{target_lang}.srt")

    generate_srt(segments, original_srt, use_translated=False)
    generate_srt(segments, translated_srt, use_translated=True)

    return original_srt, translated_srt
