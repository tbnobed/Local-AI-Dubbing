import { useState, useEffect, useCallback } from "react";
import { listJobs } from "../lib/api";
import type { Job } from "../types";

export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      const data = await listJobs();
      setJobs(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
    const interval = setInterval(fetch, 5000);
    return () => clearInterval(interval);
  }, [fetch]);

  const updateJob = useCallback((updated: Partial<Job> & { id: string }) => {
    setJobs((prev) =>
      prev.map((j) => (j.id === updated.id ? { ...j, ...updated } : j))
    );
  }, []);

  const addJob = useCallback((job: Job) => {
    setJobs((prev) => [job, ...prev]);
  }, []);

  return { jobs, loading, error, refresh: fetch, updateJob, addJob };
}
