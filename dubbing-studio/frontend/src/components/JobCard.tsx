import React, { useCallback } from "react";
import {
  FileVideo, Clock, Mic2, Download, XCircle, CheckCircle,
  AlertCircle, Loader2, ChevronRight, Users, Timer, RefreshCw
} from "lucide-react";
import { getDownloadUrl, cancelJob, retryJob } from "../lib/api";
import { useJobWebSocket } from "../hooks/useJobWebSocket";
import type { Job, WSMessage } from "../types";
import clsx from "clsx";

const STAGE_LABELS: Record<string, string> = {
  pending: "Queued",
  extracting_audio: "Extracting Audio",
  transcribing: "Transcribing Speech",
  diarizing: "Identifying Speakers",
  translating: "Translating",
  exporting_subtitles: "Exporting Subtitles",
  cloning_voices: "Preparing Voice Clones",
  synthesizing: "Synthesizing Speech",
  mixing: "Mixing Audio",
  completed: "Complete",
  failed: "Failed",
  cancelled: "Cancelled",
};

interface Props {
  job: Job;
  onUpdate: (updated: Partial<Job> & { id: string }) => void;
}

export function JobCard({ job, onUpdate }: Props) {
  const isActive = !["completed", "failed", "cancelled"].includes(job.status);

  const handleWsUpdate = useCallback(
    (msg: WSMessage) => {
      onUpdate({
        id: msg.job_id,
        status: msg.status,
        progress: msg.progress,
        current_stage: msg.stage,
        ...(msg.output_video_path ? { output_video_path: msg.output_video_path } : {}),
        ...(msg.error ? { error_message: msg.error } : {}),
      });
    },
    [onUpdate]
  );

  useJobWebSocket(job.id, { onUpdate: handleWsUpdate, enabled: isActive });

  const handleCancel = async () => {
    try {
      await cancelJob(job.id);
      onUpdate({ id: job.id, status: "cancelled" });
    } catch {}
  };

  const handleRetry = async () => {
    try {
      const updated = await retryJob(job.id);
      onUpdate({ ...updated });
    } catch (e: any) {
      alert(e.message || "Failed to retry job");
    }
  };

  const langs = `${job.source_language.toUpperCase()} → ${job.target_language.toUpperCase()}`;
  const duration = job.duration_seconds
    ? `${Math.floor(job.duration_seconds / 60)}m ${Math.floor(job.duration_seconds % 60)}s`
    : null;
  const processingTime = job.processing_time_seconds
    ? `${Math.round(job.processing_time_seconds)}s`
    : null;

  return (
    <div className={clsx(
      "glass-card p-5 transition-all duration-300 animate-slide-up",
      job.status === "completed" && "ring-1 ring-emerald-700/20",
      job.status === "failed" && "ring-1 ring-red-700/20",
    )}>
      <div className="flex items-start gap-4">
        {/* Icon */}
        <div className={clsx(
          "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5",
          job.status === "completed" ? "bg-emerald-900/50 text-emerald-400" :
          job.status === "failed" ? "bg-red-900/50 text-red-400" :
          job.status === "cancelled" ? "bg-slate-700 text-slate-500" :
          "bg-brand-900/50 text-brand-400"
        )}>
          {job.status === "completed" ? <CheckCircle className="w-5 h-5" /> :
           job.status === "failed" ? <AlertCircle className="w-5 h-5" /> :
           job.status === "cancelled" ? <XCircle className="w-5 h-5" /> :
           isActive ? <Loader2 className="w-5 h-5 animate-spin" /> :
           <FileVideo className="w-5 h-5" />}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-1">
            <p className="text-slate-200 font-medium truncate text-sm">{job.original_filename}</p>
            <StatusBadge status={job.status} />
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500 mb-3">
            <span className="font-mono text-slate-400">{langs}</span>
            {duration && (
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {duration}
              </span>
            )}
            {job.speakers_detected > 0 && (
              <span className="flex items-center gap-1">
                <Users className="w-3 h-3" />
                {job.speakers_detected} speaker{job.speakers_detected !== 1 ? "s" : ""}
              </span>
            )}
            {processingTime && (
              <span className="flex items-center gap-1">
                <Timer className="w-3 h-3" />
                processed in {processingTime}
              </span>
            )}
          </div>

          {/* Progress bar */}
          {isActive && (
            <div className="mb-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-slate-400">{STAGE_LABELS[job.current_stage] || job.current_stage}</span>
                <span className="text-xs font-mono text-brand-400">{Math.round(job.progress)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${job.progress}%` }} />
              </div>
            </div>
          )}

          {/* Error + Retry */}
          {job.status === "failed" && job.error_message && (
            <p className="text-xs text-red-400 bg-red-950/30 rounded-lg px-3 py-2 mb-3 font-mono">
              {job.error_message}
            </p>
          )}

          {(job.status === "failed" || job.status === "cancelled") && (
            <button
              onClick={handleRetry}
              className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                         bg-brand-900/50 text-brand-300 border border-brand-700/30
                         hover:bg-brand-800/50 transition-colors mb-2"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry
            </button>
          )}

          {/* Actions */}
          {job.status === "completed" && (
            <div className="flex flex-wrap gap-2 mt-2">
              {job.output_video_path && (
                <a
                  href={getDownloadUrl(job.id, "video")}
                  download
                  className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                             bg-brand-900/50 text-brand-300 border border-brand-700/30
                             hover:bg-brand-800/50 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Dubbed Video
                </a>
              )}
              {job.output_srt_path && (
                <a
                  href={getDownloadUrl(job.id, "srt", "translated")}
                  download
                  className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                             bg-brand-900/50 text-brand-300 border border-brand-700/30
                             hover:bg-brand-800/50 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  SRT ({job.target_language.toUpperCase()})
                </a>
              )}
              {job.output_srt_path && (
                <a
                  href={getDownloadUrl(job.id, "srt", "original")}
                  download
                  className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                             bg-surface-700 text-slate-300 border border-surface-500
                             hover:bg-surface-600 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  SRT ({job.source_language.toUpperCase()})
                </a>
              )}
              {job.output_vtt_path && (
                <a
                  href={getDownloadUrl(job.id, "vtt", "translated")}
                  download
                  className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                             bg-brand-900/50 text-brand-300 border border-brand-700/30
                             hover:bg-brand-800/50 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  VTT ({job.target_language.toUpperCase()})
                </a>
              )}
              {job.output_vtt_path && (
                <a
                  href={getDownloadUrl(job.id, "vtt", "original")}
                  download
                  className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                             bg-surface-700 text-slate-300 border border-surface-500
                             hover:bg-surface-600 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  VTT ({job.source_language.toUpperCase()})
                </a>
              )}
            </div>
          )}

          {isActive && job.status !== "pending" && (
            <button
              onClick={handleCancel}
              className="mt-2 text-xs text-slate-500 hover:text-red-400 transition-colors flex items-center gap-1"
            >
              <XCircle className="w-3.5 h-3.5" />
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "completed" ? "badge-completed" :
    status === "failed" ? "badge-failed" :
    status === "cancelled" ? "badge-pending" :
    "badge-running";

  const dot =
    status === "completed" ? "bg-emerald-400" :
    status === "failed" ? "bg-red-400" :
    status === "cancelled" ? "bg-slate-500" :
    "bg-brand-400 animate-pulse";

  return (
    <span className={cls}>
      <span className={clsx("w-1.5 h-1.5 rounded-full", dot)} />
      {STAGE_LABELS[status] || status}
    </span>
  );
}
