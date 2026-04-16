"""
Speaker diarization - now handled by WhisperX (pyannote community-1 under the hood).

This module provides utility functions for extracting voice samples
from diarized segments. The actual diarization is done inside
TranscriptionService.transcribe_and_diarize().
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_speaker_voice_samples(
    audio_path: str,
    segments: list,
    output_dir: str,
    min_duration: float = 8.0,
    max_duration: float = 30.0,
    vocals_path: str | None = None,
) -> dict[str, str]:
    """
    Extract audio samples for each speaker from diarized segments.

    If vocals_path is provided (from Demucs separation), extracts from
    the clean vocal stem for much better voice cloning quality.
    Otherwise falls back to the original mixed audio.

    Returns dict of speaker_id -> audio_file_path.
    """
    import ffmpeg

    source_audio = vocals_path if vocals_path and os.path.exists(vocals_path) else audio_path
    if source_audio != audio_path:
        logger.info(f"Extracting voice samples from clean vocal stem")
    else:
        logger.info(f"Extracting voice samples from original audio")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_segments: dict[str, list] = {}
    for seg in segments:
        speaker = getattr(seg, "speaker", "SPEAKER_00") or "SPEAKER_00"
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(seg)

    samples = {}
    for speaker, segs in speaker_segments.items():
        sorted_segs = sorted(segs, key=lambda s: s.end - s.start, reverse=True)

        accumulated = 0.0
        selected = []
        for seg in sorted_segs:
            dur = seg.end - seg.start
            if dur < 0.5:
                continue
            selected.append(seg)
            accumulated += dur
            if accumulated >= max_duration:
                break
            if len(selected) >= 5:
                break

        if not selected:
            continue

        if accumulated < min_duration:
            logger.warning(
                f"Speaker {speaker}: only {accumulated:.1f}s of audio "
                f"(< {min_duration}s minimum), voice cloning quality may be reduced"
            )

        selected.sort(key=lambda s: s.start)
        sample_path = str(output_dir / f"{speaker}_sample.wav")

        try:
            if len(selected) == 1:
                seg = selected[0]
                (
                    ffmpeg
                    .input(source_audio, ss=seg.start, t=seg.end - seg.start)
                    .output(sample_path, ar=22050, ac=1, acodec="pcm_s16le")
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                clip_paths = []
                for i, seg in enumerate(selected):
                    clip_path = str(output_dir / f"{speaker}_clip_{i}.wav")
                    (
                        ffmpeg
                        .input(source_audio, ss=seg.start, t=seg.end - seg.start)
                        .output(clip_path, ar=22050, ac=1, acodec="pcm_s16le")
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    clip_paths.append(clip_path)

                concat_file = str(output_dir / f"{speaker}_concat.txt")
                with open(concat_file, "w") as f:
                    for cp in clip_paths:
                        f.write(f"file '{cp}'\n")

                (
                    ffmpeg
                    .input(concat_file, f="concat", safe=0)
                    .output(sample_path, ar=22050, ac=1, acodec="pcm_s16le")
                    .overwrite_output()
                    .run(quiet=True)
                )

                for cp in clip_paths:
                    try:
                        os.remove(cp)
                    except OSError:
                        pass
                try:
                    os.remove(concat_file)
                except OSError:
                    pass

            samples[speaker] = sample_path
            logger.info(f"Extracted {accumulated:.1f}s voice sample for {speaker}")

        except Exception as e:
            logger.error(f"Failed to extract voice sample for {speaker}: {e}")

    return samples
