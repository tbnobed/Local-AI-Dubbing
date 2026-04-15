"""
Speaker diarization service using pyannote.audio.
Identifies who is speaking when in the audio.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    num_speakers: int


class DiarizationService:
    def __init__(self, config):
        self.config = config
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            import torch

            logger.info("Loading diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                self.config.diarization_model,
                use_auth_token=self.config.hf_token,
            )

            if torch.cuda.is_available() and self.config.use_gpu:
                device_id = self.config.secondary_gpu_id
                self._pipeline = self._pipeline.to(torch.device(f"cuda:{device_id}"))
                logger.info(f"Diarization pipeline on cuda:{device_id}")

        return self._pipeline

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        pipeline = self._load_pipeline()
        logger.info(f"Diarizing: {audio_path}")

        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers

        diarization = pipeline(audio_path, **kwargs)

        turns = []
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(SpeakerTurn(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))
            speakers.add(speaker)

        logger.info(f"Diarization complete: {len(turns)} turns, {len(speakers)} speakers")

        return DiarizationResult(turns=turns, num_speakers=len(speakers))

    def assign_speakers_to_segments(self, segments, diarization: DiarizationResult):
        """
        Match transcription segments to detected speakers by overlap.
        """
        for segment in segments:
            best_speaker = None
            best_overlap = 0.0

            for turn in diarization.turns:
                overlap_start = max(segment.start, turn.start)
                overlap_end = min(segment.end, turn.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker

            segment.speaker = best_speaker or "SPEAKER_00"

        return segments

    def extract_speaker_voice_samples(
        self,
        audio_path: str,
        diarization: DiarizationResult,
        output_dir: str,
        min_duration: float = 8.0,
        max_duration: float = 30.0,
    ) -> dict[str, str]:
        """
        Extract audio samples for each speaker (for voice cloning reference).
        Returns dict of speaker_id -> audio_file_path.
        """
        import ffmpeg
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        speaker_turns: dict[str, list[SpeakerTurn]] = {}
        for turn in diarization.turns:
            if turn.speaker not in speaker_turns:
                speaker_turns[turn.speaker] = []
            speaker_turns[turn.speaker].append(turn)

        samples = {}
        for speaker, turns in speaker_turns.items():
            # Sort by length, pick longest continuous segments up to max_duration
            sorted_turns = sorted(turns, key=lambda t: t.end - t.start, reverse=True)
            accumulated = 0.0
            selected_turns = []

            for turn in sorted_turns:
                dur = turn.end - turn.start
                if dur < 1.0:
                    continue
                selected_turns.append(turn)
                accumulated += dur
                if accumulated >= max_duration:
                    break

            if accumulated < min_duration:
                logger.warning(f"Speaker {speaker} has only {accumulated:.1f}s of audio, may affect cloning quality")

            selected_turns.sort(key=lambda t: t.start)

            if not selected_turns:
                continue

            sample_path = str(output_dir / f"{speaker}_sample.wav")

            # Build filter_complex to concatenate clips
            segments_args = []
            for i, turn in enumerate(selected_turns):
                segments_args.append({
                    "start": turn.start,
                    "end": turn.end,
                    "index": i,
                })

            try:
                if len(segments_args) == 1:
                    t = segments_args[0]
                    (
                        ffmpeg
                        .input(audio_path, ss=t["start"], t=t["end"] - t["start"])
                        .output(sample_path, ar=22050, ac=1, acodec="pcm_s16le")
                        .overwrite_output()
                        .run(quiet=True)
                    )
                else:
                    inputs = []
                    filter_parts = []
                    for i, t in enumerate(segments_args):
                        inp = ffmpeg.input(audio_path, ss=t["start"], t=t["end"] - t["start"])
                        inputs.append(inp)
                        filter_parts.append(f"[{i}:a]")

                    concat_str = "".join(filter_parts) + f"concat=n={len(inputs)}:v=0:a=1[out]"
                    (
                        ffmpeg
                        .output(*inputs, sample_path, ar=22050, ac=1, acodec="pcm_s16le",
                                filter_complex=concat_str, map="[out]")
                        .overwrite_output()
                        .run(quiet=True)
                    )

                samples[speaker] = sample_path
                logger.info(f"Extracted {accumulated:.1f}s voice sample for {speaker}")

            except Exception as e:
                logger.error(f"Failed to extract voice sample for {speaker}: {e}")

        return samples

    def unload(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
