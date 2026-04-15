import React, { useState, useEffect } from "react";
import { Cpu, Server, Wifi, WifiOff, Thermometer } from "lucide-react";
import { getSystemStatus } from "../lib/api";
import type { SystemStatus } from "../types";
import clsx from "clsx";

export function SystemStatusBar() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetch = async () => {
      try {
        const s = await getSystemStatus();
        setStatus(s);
        setError(false);
      } catch {
        setError(true);
      }
    };
    fetch();
    const interval = setInterval(fetch, 10000);
    return () => clearInterval(interval);
  }, []);

  if (!status && !error) return null;

  return (
    <div className="flex items-center gap-4 flex-wrap">
      {/* API status */}
      <div className="flex items-center gap-1.5 text-xs">
        {error ? (
          <>
            <WifiOff className="w-3.5 h-3.5 text-red-400" />
            <span className="text-red-400">API Offline</span>
          </>
        ) : (
          <>
            <Wifi className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">Connected</span>
          </>
        )}
      </div>

      {status && (
        <>
          {/* Redis */}
          <div className="flex items-center gap-1.5 text-xs">
            <div className={clsx(
              "w-1.5 h-1.5 rounded-full",
              status.redis_connected ? "bg-emerald-400" : "bg-red-400"
            )} />
            <span className="text-slate-400">Redis</span>
          </div>

          {/* Workers */}
          <div className="flex items-center gap-1.5 text-xs">
            <Server className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-400">{status.celery_workers} worker{status.celery_workers !== 1 ? "s" : ""}</span>
          </div>

          {/* GPUs */}
          {status.gpus.map((gpu) => (
            <div key={gpu.id} className="flex items-center gap-2 text-xs">
              <Cpu className="w-3.5 h-3.5 text-brand-400" />
              <span className="text-slate-400 max-w-[140px] truncate">{gpu.name}</span>
              <div className="flex items-center gap-1">
                <div className="w-16 h-1.5 bg-surface-600 rounded-full overflow-hidden">
                  <div
                    className={clsx(
                      "h-full rounded-full transition-all duration-500",
                      gpu.utilization_percent > 90 ? "bg-red-500" :
                      gpu.utilization_percent > 60 ? "bg-amber-500" :
                      "bg-brand-500"
                    )}
                    style={{ width: `${Math.min(100, gpu.utilization_percent)}%` }}
                  />
                </div>
                <span className="text-slate-500 font-mono w-8">{Math.round(gpu.utilization_percent)}%</span>
              </div>
              {gpu.temperature_c !== null && (
                <span className={clsx(
                  "flex items-center gap-0.5",
                  gpu.temperature_c > 85 ? "text-red-400" :
                  gpu.temperature_c > 70 ? "text-amber-400" :
                  "text-slate-500"
                )}>
                  <Thermometer className="w-3 h-3" />
                  {Math.round(gpu.temperature_c)}°C
                </span>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
