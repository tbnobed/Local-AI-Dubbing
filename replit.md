# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.
Also contains the `dubbing-studio/` project — a complete local AI video dubbing system.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

---

## DubbingStudio (dubbing-studio/)

A standalone local AI video dubbing system. Runs entirely on local GPU hardware.

### Architecture
- **Backend**: Python + FastAPI + Celery + Redis
- **Frontend**: React + Vite + Tailwind CSS
- **Pipeline**: faster-whisper → pyannote → NLLB-200 → XTTS v2 → ffmpeg
- **DB**: SQLite (local jobs tracking)

### Key Files
- `dubbing-studio/README.md` — full setup and usage guide
- `dubbing-studio/backend/app/` — FastAPI app, services, workers
- `dubbing-studio/frontend/src/` — React UI
- `dubbing-studio/scripts/` — setup.sh, start.sh, stop.sh

### Models Used
- Whisper large-v3 (transcription)
- pyannote/speaker-diarization-3.1 (speaker detection)
- facebook/nllb-200-distilled-600M (translation)
- tts_models/multilingual/multi-dataset/xtts_v2 (voice cloning TTS)
