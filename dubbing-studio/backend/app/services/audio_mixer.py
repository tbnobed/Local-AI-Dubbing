"""
Audio mixing service - assembles synthesized speech segments back into
a complete audio track and mixes it with the original video.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioMixerService:
    def __init__(self, config):
        self.config = config

    def extract_audio(self, video_path: str, output_path: str) -> str:
        import ffmpeg

        logger.info(f"Extracting audio from {video_path}")
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ar=22050, ac=1, acodec="pcm_s16le", vn=None)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path

    def build_dubbed_audio(
        self,
        segments: list,
        total_duration: float,
        output_path: str,
        sample_rate: int = 44100,
    ) -> str:
        """
        Construct a full-length audio track by placing synthesized segments
        at their original timestamps.
        """
        import numpy as np
        import soundfile as sf

        logger.info(f"Building dubbed audio track ({total_duration:.1f}s)")

        total_samples = int(total_duration * sample_rate) + sample_rate
        output_audio = np.zeros(total_samples, dtype=np.float32)

        for seg in segments:
            if not getattr(seg, "synth_audio_path", None):
                continue
            if not os.path.exists(seg.synth_audio_path):
                continue

            try:
                audio, sr = sf.read(seg.synth_audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Resample if needed
                if sr != sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

                start_sample = int(seg.start * sample_rate)
                end_sample = start_sample + len(audio)

                if end_sample > len(output_audio):
                    audio = audio[: len(output_audio) - start_sample]
                    end_sample = len(output_audio)

                output_audio[start_sample:end_sample] += audio

            except Exception as e:
                logger.error(f"Failed to mix segment at {seg.start:.2f}s: {e}")

        # Normalize
        peak = np.max(np.abs(output_audio))
        if peak > 1.0:
            output_audio = output_audio / peak * 0.95

        sf.write(output_path, output_audio, sample_rate)
        logger.info(f"Dubbed audio written to {output_path}")
        return output_path

    def build_final_audio(
        self,
        dubbed_vocals_path: str,
        instrumental_path: str | None,
        output_path: str,
        instrumental_volume: float = 1.0,
        vocals_volume: float = 1.0,
    ) -> str:
        """
        Mix dubbed vocals with the original instrumental stem from Demucs.
        If no instrumental stem is available, returns just the dubbed vocals.
        """
        import ffmpeg

        if not instrumental_path or not os.path.exists(instrumental_path):
            logger.info("No instrumental stem — using dubbed vocals only")
            shutil.copy2(dubbed_vocals_path, output_path)
            return output_path

        logger.info(f"Mixing dubbed vocals (vol={vocals_volume}) + instrumentals (vol={instrumental_volume})")

        vocals_in = ffmpeg.input(dubbed_vocals_path).audio
        instrumental_in = ffmpeg.input(instrumental_path).audio

        if vocals_volume != 1.0:
            vocals_in = vocals_in.filter("volume", vocals_volume)
        if instrumental_volume != 1.0:
            instrumental_in = instrumental_in.filter("volume", instrumental_volume)

        mixed = ffmpeg.filter(
            [vocals_in, instrumental_in],
            "amix",
            inputs=2,
            duration="longest",
            normalize=0,
        )

        ffmpeg.output(mixed, output_path, acodec="pcm_s16le", ar=44100, ac=1).overwrite_output().run(quiet=True)
        logger.info(f"Final mixed audio written: {output_path}")
        return output_path

    def merge_audio_into_video(
        self,
        video_path: str,
        dubbed_audio_path: str,
        output_path: str,
        original_audio_volume: float = 0.0,
    ) -> str:
        """
        Replace video audio with dubbed track.
        Optionally keep a quiet version of original audio underneath.
        """
        import ffmpeg

        logger.info(f"Merging dubbed audio into video")

        video_in = ffmpeg.input(video_path)
        dubbed_in = ffmpeg.input(dubbed_audio_path)

        if original_audio_volume > 0:
            original_audio = video_in.audio.filter("volume", original_audio_volume)
            mixed = ffmpeg.filter([dubbed_in.audio, original_audio], "amix", inputs=2, duration="first")
            output = ffmpeg.output(
                video_in.video, mixed, output_path,
                vcodec="copy", acodec="aac", audio_bitrate="192k",
                shortest=None,
            )
        else:
            output = ffmpeg.output(
                video_in.video, dubbed_in.audio, output_path,
                vcodec="copy", acodec="aac", audio_bitrate="192k",
                shortest=None,
            )

        output.overwrite_output().run(quiet=True)
        logger.info(f"Final dubbed video: {output_path}")
        return output_path

    def get_video_duration(self, video_path: str) -> float:
        import ffmpeg

        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
        return duration

    def get_video_info(self, video_path: str) -> dict:
        import ffmpeg

        probe = ffmpeg.probe(video_path)
        info = {
            "duration": float(probe["format"].get("duration", 0)),
            "size_bytes": int(probe["format"].get("size", 0)),
            "format": probe["format"].get("format_name", ""),
        }

        for stream in probe.get("streams", []):
            if stream["codec_type"] == "video":
                info["video_codec"] = stream.get("codec_name", "")
                info["width"] = stream.get("width", 0)
                info["height"] = stream.get("height", 0)
                info["fps"] = stream.get("r_frame_rate", "")
            elif stream["codec_type"] == "audio":
                info["audio_codec"] = stream.get("codec_name", "")
                info["sample_rate"] = stream.get("sample_rate", "")
                info["channels"] = stream.get("channels", 0)

        return info
