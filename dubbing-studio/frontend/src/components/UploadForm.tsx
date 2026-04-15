import React, { useState, useRef, useCallback } from "react";
import { Upload, FileVideo, X, Mic, Languages, Subtitles } from "lucide-react";
import { createJob } from "../lib/api";
import type { Job } from "../types";
import clsx from "clsx";

const LANGUAGES = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ja", name: "Japanese" },
  { code: "zh", name: "Chinese" },
  { code: "ko", name: "Korean" },
  { code: "ar", name: "Arabic" },
  { code: "ru", name: "Russian" },
  { code: "hi", name: "Hindi" },
];

const ACCEPTED = ".mp4,.mkv,.avi,.mov,.webm,.m4v,.flv,.wmv";

interface Props {
  onJobCreated: (job: Job) => void;
}

export function UploadForm({ onJobCreated }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("es");
  const [voiceCloning, setVoiceCloning] = useState(true);
  const [exportSrt, setExportSrt] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setError(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    if (sourceLang === targetLang) {
      setError("Source and target languages must be different.");
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("source_language", sourceLang);
    formData.append("target_language", targetLang);
    formData.append("voice_cloning_enabled", String(voiceCloning));
    formData.append("export_srt", String(exportSrt));

    try {
      const job = await createJob(formData);
      onJobCreated(job);
      setFile(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes > 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes > 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    return `${(bytes / 1e3).toFixed(0)} KB`;
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Drop zone */}
      <div
        className={clsx(
          "relative rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer",
          dragging
            ? "border-brand-400 bg-brand-950/30"
            : file
            ? "border-emerald-600/50 bg-emerald-950/10"
            : "border-surface-400 bg-surface-800/50 hover:border-brand-600/60 hover:bg-surface-700/30"
        )}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !file && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />

        <div className="p-8 text-center">
          {file ? (
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-emerald-900/50 border border-emerald-700/30 flex items-center justify-center flex-shrink-0">
                <FileVideo className="w-6 h-6 text-emerald-400" />
              </div>
              <div className="flex-1 text-left min-w-0">
                <p className="text-slate-200 font-medium truncate">{file.name}</p>
                <p className="text-slate-500 text-sm mt-0.5">{formatSize(file.size)}</p>
              </div>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setFile(null); }}
                className="w-8 h-8 rounded-lg bg-surface-600 hover:bg-surface-500 flex items-center justify-center transition-colors flex-shrink-0"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
          ) : (
            <>
              <div className="w-14 h-14 rounded-2xl bg-surface-700 border border-surface-500 flex items-center justify-center mx-auto mb-4">
                <Upload className="w-7 h-7 text-brand-400" />
              </div>
              <p className="text-slate-200 font-medium mb-1">Drop your video here</p>
              <p className="text-slate-500 text-sm">
                or <span className="text-brand-400 hover:text-brand-300 cursor-pointer">browse files</span>
              </p>
              <p className="text-slate-600 text-xs mt-3">MP4, MKV, AVI, MOV, WebM — up to 10 GB</p>
            </>
          )}
        </div>
      </div>

      {/* Language selects */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <Languages className="w-4 h-4 text-slate-500" />
            Source Language
          </label>
          <div className="relative">
            <select
              value={sourceLang}
              onChange={(e) => setSourceLang(e.target.value)}
              className="select-field w-full pr-8"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>{l.name}</option>
              ))}
            </select>
            <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2">
              <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <Languages className="w-4 h-4 text-slate-500" />
            Target Language
          </label>
          <div className="relative">
            <select
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
              className="select-field w-full pr-8"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>{l.name}</option>
              ))}
            </select>
            <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2">
              <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Options */}
      <div className="space-y-3">
        <Toggle
          enabled={voiceCloning}
          onChange={setVoiceCloning}
          icon={<Mic className="w-4 h-4" />}
          label="Voice Cloning"
          description="Replicate each speaker's voice in the target language"
        />
        <Toggle
          enabled={exportSrt}
          onChange={setExportSrt}
          icon={<SubtitlesIcon />}
          label="Export SRT Subtitles"
          description="Generate subtitle files for both original and translated audio"
        />
      </div>

      {error && (
        <div className="rounded-xl bg-red-950/50 border border-red-800/40 px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={!file || uploading}
        className="btn-primary w-full flex items-center justify-center gap-2 py-3"
      >
        {uploading ? (
          <>
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Uploading...
          </>
        ) : (
          <>
            <Upload className="w-4 h-4" />
            Start Dubbing
          </>
        )}
      </button>
    </form>
  );
}

function Toggle({
  enabled,
  onChange,
  icon,
  label,
  description,
}: {
  enabled: boolean;
  onChange: (v: boolean) => void;
  icon: React.ReactNode;
  label: string;
  description: string;
}) {
  return (
    <div
      className={clsx(
        "flex items-center gap-4 p-4 rounded-xl border cursor-pointer transition-all duration-200",
        enabled
          ? "bg-brand-950/40 border-brand-700/40"
          : "bg-surface-800/50 border-surface-600/50 hover:border-surface-500"
      )}
      onClick={() => onChange(!enabled)}
    >
      <div className={clsx(
        "w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors",
        enabled ? "bg-brand-700/50 text-brand-300" : "bg-surface-700 text-slate-500"
      )}>
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-200">{label}</p>
        <p className="text-xs text-slate-500 mt-0.5">{description}</p>
      </div>
      <div className={clsx(
        "w-11 h-6 rounded-full flex items-center transition-all duration-300 flex-shrink-0",
        enabled ? "bg-brand-600" : "bg-surface-500"
      )}>
        <div className={clsx(
          "w-5 h-5 rounded-full bg-white shadow transition-transform duration-300",
          enabled ? "translate-x-5.5" : "translate-x-0.5"
        )} style={{ transform: enabled ? "translateX(22px)" : "translateX(2px)" }} />
      </div>
    </div>
  );
}

function SubtitlesIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  );
}
