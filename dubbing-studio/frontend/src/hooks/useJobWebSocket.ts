import { useEffect, useRef, useCallback } from "react";
import { createJobWebSocket } from "../lib/api";
import type { WSMessage, Job } from "../types";

interface Options {
  onUpdate: (msg: WSMessage) => void;
  enabled: boolean;
}

export function useJobWebSocket(jobId: string, { onUpdate, enabled }: Options) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!enabled || !mountedRef.current) return;

    const ws = createJobWebSocket(jobId);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage;
        if (msg.job_id) {
          onUpdate(msg);
        }
      } catch {}
    };

    ws.onerror = () => {};

    ws.onclose = () => {
      if (!mountedRef.current) return;
      reconnectRef.current = setTimeout(connect, 2000);
    };
  }, [jobId, enabled, onUpdate]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);
}
