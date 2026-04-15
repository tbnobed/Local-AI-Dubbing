import type { Job, SystemStatus } from "../types";

const BASE = "/api";

export async function createJob(formData: FormData): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/`, { method: "POST", body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to create job");
  }
  return res.json();
}

export async function listJobs(): Promise<Job[]> {
  const res = await fetch(`${BASE}/jobs/`);
  if (!res.ok) throw new Error("Failed to fetch jobs");
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${BASE}/jobs/${jobId}`, { method: "DELETE" });
}

export async function retryJob(jobId: string): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/${jobId}/retry`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to retry job");
  }
  return res.json();
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${BASE}/system/status`);
  if (!res.ok) throw new Error("Failed to fetch system status");
  return res.json();
}

export async function getLanguages(): Promise<Record<string, { name: string; code: string }>> {
  const res = await fetch(`${BASE}/system/languages`);
  if (!res.ok) throw new Error("Failed to fetch languages");
  return res.json();
}

export async function clearAllJobs(): Promise<void> {
  const res = await fetch(`${BASE}/jobs/`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to clear jobs");
}

export function getDownloadUrl(jobId: string, type: "video" | "srt" | "vtt", lang?: string): string {
  if (type === "video") return `${BASE}/jobs/${jobId}/download/video`;
  if (type === "vtt") return `${BASE}/jobs/${jobId}/download/vtt?lang=${lang || "translated"}`;
  return `${BASE}/jobs/${jobId}/download/srt?lang=${lang || "translated"}`;
}

export function createJobWebSocket(jobId: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${proto}//${host}/ws/jobs/${jobId}`);
}
