export type JobStatus =
  | "pending"
  | "extracting_audio"
  | "separating_audio"
  | "transcribing"
  | "diarizing"
  | "translating"
  | "exporting_subtitles"
  | "cloning_voices"
  | "synthesizing"
  | "mixing"
  | "completed"
  | "failed"
  | "cancelled";

export interface Job {
  id: string;
  status: JobStatus;
  progress: number;
  current_stage: string;
  original_filename: string;
  source_language: string;
  target_language: string;
  speakers_detected: number;
  duration_seconds: number | null;
  output_video_path: string | null;
  output_srt_path: string | null;
  output_original_srt_path: string | null;
  output_vtt_path: string | null;
  output_original_vtt_path: string | null;
  error_message: string | null;
  processing_time_seconds: number | null;
  created_at: string;
  transcription_data: {
    language: string;
    language_probability: number;
    segments_count: number;
    duration: number;
  } | null;
}

export interface Language {
  code: string;
  name: string;
}

export interface GPUInfo {
  id: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  utilization_percent: number;
  temperature_c: number | null;
}

export interface SystemStatus {
  gpus: GPUInfo[];
  redis_connected: boolean;
  celery_workers: number;
  supported_languages: Record<string, { name: string; nllb: string; whisper: string }>;
}

export interface WSMessage {
  job_id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  error?: string;
  output_video_path?: string;
}
