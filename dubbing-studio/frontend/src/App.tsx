import React from "react";
import { Mic2, Layers, Trash2 } from "lucide-react";
import { UploadForm } from "./components/UploadForm";
import { JobCard } from "./components/JobCard";
import { SystemStatusBar } from "./components/SystemStatusBar";
import { useJobs } from "./hooks/useJobs";
import type { Job } from "./types";

export default function App() {
  const { jobs, loading, updateJob, addJob, clearAll } = useJobs();

  const handleJobCreated = (job: Job) => {
    addJob(job);
  };

  const activeJobs = jobs.filter(j => !["completed", "failed", "cancelled"].includes(j.status));
  const completedJobs = jobs.filter(j => j.status === "completed");
  const failedJobs = jobs.filter(j => ["failed", "cancelled"].includes(j.status));

  return (
    <div className="min-h-screen bg-surface-900">
      {/* Header */}
      <header className="border-b border-surface-700/60 bg-surface-800/60 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-600 to-brand-800 flex items-center justify-center shadow-lg shadow-brand-950/50">
              <Mic2 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-slate-100 font-semibold text-sm leading-tight">DubbingStudio</h1>
              <p className="text-slate-500 text-xs">AI Video Dubbing</p>
            </div>
          </div>

          <SystemStatusBar />
        </div>
      </header>

      {/* Main layout */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[420px_1fr] gap-8 items-start">

          {/* Left: Upload panel */}
          <div className="space-y-4">
            <div>
              <h2 className="text-slate-100 font-semibold text-lg mb-1">New Dubbing Job</h2>
              <p className="text-slate-500 text-sm">
                Upload a video to transcribe, translate, and dub with voice cloning.
              </p>
            </div>

            <div className="glass-card p-6">
              <UploadForm onJobCreated={handleJobCreated} />
            </div>

            {/* Pipeline info */}
            <div className="glass-card p-5">
              <h3 className="text-slate-300 font-medium text-sm mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4 text-brand-400" />
                Pipeline Stages
              </h3>
              <div className="space-y-2">
                {[
                  { label: "Audio Extraction", desc: "ffmpeg" },
                  { label: "Speech Transcription", desc: "Whisper large-v3-turbo" },
                  { label: "Speaker Diarization", desc: "pyannote community-1" },
                  { label: "Translation", desc: "NLLB-200 3.3B (Meta)" },
                  { label: "Voice Cloning + TTS", desc: "Fish Speech 1.5" },
                  { label: "Audio Mixing", desc: "ffmpeg + librosa" },
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs">
                    <div className="w-5 h-5 rounded-full bg-surface-600 border border-surface-500 flex items-center justify-center flex-shrink-0">
                      <span className="text-slate-500 text-[10px] font-mono">{i + 1}</span>
                    </div>
                    <div className="flex-1 flex items-center justify-between gap-2">
                      <span className="text-slate-300">{step.label}</span>
                      <span className="text-slate-600 font-mono">{step.desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Jobs list */}
          <div className="space-y-6">
            {jobs.length > 0 && (
              <div className="flex justify-end">
                <button
                  onClick={() => { if (confirm("Clear all jobs?")) clearAll(); }}
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-red-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-surface-700/50"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear All
                </button>
              </div>
            )}

            {/* Active jobs */}
            {activeJobs.length > 0 && (
              <section>
                <h2 className="text-slate-300 font-semibold text-sm mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-brand-400 animate-pulse" />
                  Processing ({activeJobs.length})
                </h2>
                <div className="space-y-3">
                  {activeJobs.map(job => (
                    <JobCard key={job.id} job={job} onUpdate={updateJob} />
                  ))}
                </div>
              </section>
            )}

            {/* Completed jobs */}
            {completedJobs.length > 0 && (
              <section>
                <h2 className="text-slate-400 font-semibold text-sm mb-3">
                  Completed ({completedJobs.length})
                </h2>
                <div className="space-y-3">
                  {completedJobs.map(job => (
                    <JobCard key={job.id} job={job} onUpdate={updateJob} />
                  ))}
                </div>
              </section>
            )}

            {/* Failed/cancelled */}
            {failedJobs.length > 0 && (
              <section>
                <h2 className="text-slate-500 font-semibold text-sm mb-3">
                  Failed / Cancelled ({failedJobs.length})
                </h2>
                <div className="space-y-3">
                  {failedJobs.map(job => (
                    <JobCard key={job.id} job={job} onUpdate={updateJob} />
                  ))}
                </div>
              </section>
            )}

            {/* Empty state */}
            {!loading && jobs.length === 0 && (
              <div className="text-center py-16">
                <div className="w-16 h-16 rounded-2xl bg-surface-800 border border-surface-600 flex items-center justify-center mx-auto mb-4">
                  <Mic2 className="w-8 h-8 text-surface-400" />
                </div>
                <p className="text-slate-400 font-medium mb-1">No jobs yet</p>
                <p className="text-slate-600 text-sm">Upload a video to get started</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
