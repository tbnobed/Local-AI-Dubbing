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
    min_duration: float = 3.0,
    max_duration: float = 8.0,
    vocals_path: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Extract audio samples for each speaker from diarized segments.

    If vocals_path is provided (from Demucs separation), extracts from
    the clean vocal stem for much better voice cloning quality.
    Otherwise falls back to the original mixed audio.

    Returns (samples, texts):
      samples: speaker_id -> audio_file_path
      texts:   speaker_id -> concatenated transcript of the selected segments
               (used as Fish Speech reference text for voice cloning)
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

    # Build a flat sorted list of (start, end, speaker) for overlap detection.
    all_ranges = sorted(
        [
            (s.start, s.end, getattr(s, "speaker", "SPEAKER_00") or "SPEAKER_00")
            for s in segments
        ],
        key=lambda r: r[0],
    )

    def _overlaps_other_speaker(seg_start: float, seg_end: float, this_speaker: str,
                                pad: float = 0.3) -> bool:
        for s, e, spk in all_ranges:
            if spk == this_speaker:
                continue
            if e + pad < seg_start:
                continue
            if s - pad > seg_end:
                break
            return True
        return False

    # Score each candidate segment for voice-cloning suitability.
    # Sweet spot for Fish Speech reference is ~5s. Avoid overlapping speech.
    # Score is RELATIVE — we don't hard-filter on duration so every speaker
    # gets some clip. The TTS worker will truncate to max_ref_seconds anyway.
    IDEAL_DUR = 5.0
    MAX_REF = 8.0

    def _quality_score(seg) -> float:
        dur = seg.end - seg.start
        if dur < 0.4:
            return -10.0  # genuinely useless
        # Bell curve around IDEAL_DUR, with a softer falloff for short clips
        if dur <= MAX_REF:
            score = 1.0 - abs(dur - IDEAL_DUR) / IDEAL_DUR
        else:
            score = max(0.1, 1.0 - (dur - MAX_REF) / 10.0)
        # Penalize segments overlapping another speaker
        if _overlaps_other_speaker(seg.start, seg.end,
                                   getattr(seg, "speaker", "SPEAKER_00")):
            score -= 0.4
        # Penalize empty/very short text (often noise/laughter)
        text = (getattr(seg, "text", "") or "").strip()
        if len(text) < 8:
            score -= 0.2
        return score

    samples = {}
    texts = {}
    for speaker, segs in speaker_segments.items():
        scored = [(s, _quality_score(s)) for s in segs if (s.end - s.start) >= 0.4]
        # Sort by quality desc; tie-break by longer duration
        scored.sort(key=lambda t: (t[1], t[0].end - t[0].start), reverse=True)

        # Pick the SINGLE highest-quality clip if it's long enough on its own.
        # If even the best is short, concatenate the top few clips so the
        # speaker still gets a usable reference rather than being dropped.
        accumulated = 0.0
        selected = []
        if scored:
            best_seg = scored[0][0]
            best_dur = best_seg.end - best_seg.start
            if best_dur >= min_duration:
                selected = [best_seg]
                accumulated = best_dur
            else:
                for seg, _score in scored:
                    selected.append(seg)
                    accumulated += seg.end - seg.start
                    if accumulated >= max_duration:
                        break
                    if len(selected) >= 4:
                        break

        # Last-resort fallback: take any clip we can find for this speaker
        if not selected:
            any_segs = [s for s in segs if (s.end - s.start) >= 0.2]
            if any_segs:
                any_segs.sort(key=lambda s: s.end - s.start, reverse=True)
                selected = any_segs[:1]
                accumulated = selected[0].end - selected[0].start

        if not selected:
            logger.warning(
                f"No usable audio segments for {speaker} — voice cloning "
                f"will fall back to another speaker for this voice"
            )
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
                    .output(sample_path, ar=44100, ac=1, acodec="pcm_s16le")
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
                        .output(clip_path, ar=44100, ac=1, acodec="pcm_s16le")
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
                    .output(sample_path, ar=44100, ac=1, acodec="pcm_s16le")
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
            ref_text = " ".join(
                (getattr(s, "text", "") or "").strip() for s in selected
            ).strip()
            texts[speaker] = ref_text
            logger.info(
                f"Extracted {accumulated:.1f}s voice sample for {speaker} "
                f"(ref text: {len(ref_text)} chars)"
            )

        except Exception as e:
            logger.error(f"Failed to extract voice sample for {speaker}: {e}")

    return samples, texts
