"""
SRT and WebVTT subtitle file generator from transcription/translation segments.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def seconds_to_srt_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    millis = int((seconds - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    millis = int((seconds - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _get_cues(segments: list, use_translated: bool = False):
    cues = []
    for seg in segments:
        if not seg.text.strip():
            continue
        text = getattr(seg, "translated_text", seg.text) if use_translated else seg.text
        text = text.strip()
        if text:
            cues.append((seg.start, seg.end, text))
    return cues


def generate_srt(segments: list, output_path: str, use_translated: bool = False) -> str:
    cues = _get_cues(segments, use_translated)
    lines = []
    for i, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_timestamp(start)} --> {seconds_to_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"SRT written: {output_path} ({len(cues)} entries)")
    return output_path


def generate_vtt(segments: list, output_path: str, use_translated: bool = False) -> str:
    cues = _get_cues(segments, use_translated)
    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{seconds_to_vtt_timestamp(start)} --> {seconds_to_vtt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"VTT written: {output_path} ({len(cues)} entries)")
    return output_path


def generate_all_subtitles(
    segments: list,
    output_dir: str,
    job_id: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_srt = str(output_dir / f"{job_id}_original_{source_lang}.srt")
    translated_srt = str(output_dir / f"{job_id}_translated_{target_lang}.srt")
    original_vtt = str(output_dir / f"{job_id}_original_{source_lang}.vtt")
    translated_vtt = str(output_dir / f"{job_id}_translated_{target_lang}.vtt")

    generate_srt(segments, original_srt, use_translated=False)
    generate_srt(segments, translated_srt, use_translated=True)
    generate_vtt(segments, original_vtt, use_translated=False)
    generate_vtt(segments, translated_vtt, use_translated=True)

    return {
        "original_srt": original_srt,
        "translated_srt": translated_srt,
        "original_vtt": original_vtt,
        "translated_vtt": translated_vtt,
    }
