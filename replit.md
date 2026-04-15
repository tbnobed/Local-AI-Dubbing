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
- **Pipeline**: WhisperX → MADLAD-400 → Fish Speech 1.5 → ffmpeg
- **DB**: SQLite (local jobs tracking)

### Key Files
- `dubbing-studio/README.md` — full setup and usage guide
- `dubbing-studio/backend/app/services/transcription.py` — WhisperX transcription + alignment + diarization
- `dubbing-studio/backend/app/services/translation.py` — MADLAD-400 translation
- `dubbing-studio/backend/app/services/tts.py` — Fish Speech voice cloning TTS
- `dubbing-studio/backend/app/services/diarization.py` — Speaker voice sample extraction
- `dubbing-studio/backend/app/services/audio_mixer.py` — Audio mixing + video assembly
- `dubbing-studio/backend/app/workers/pipeline.py` — Celery pipeline orchestrator
- `dubbing-studio/backend/app/config.py` — All model/GPU settings
- `dubbing-studio/frontend/src/` — React UI
- `dubbing-studio/scripts/` — setup.sh, start.sh, stop.sh

### Models Used (2025 Upgrade)
- **Whisper large-v3-turbo via transformers** (transcription — pure PyTorch, Blackwell-safe)
- **WhisperX** (word-level alignment via wav2vec2 + speaker diarization via pyannote)
- **pyannote community-1** (speaker diarization, requires HF token)
- **MADLAD-400 3B** by Google (translation, Apache 2.0 license, 400+ languages)
- **Fish Speech 1.5** (voice cloning TTS, 80+ languages, CUDA 12.x native)

### GPU Setup
- Designed for dual-GPU: RTX 5090 (primary) + RTX PRO 4500 Blackwell (secondary)
- Both GPUs are Blackwell sm_120 — requires PyTorch cu128
- **CTranslate2/faster-whisper DO NOT support sm_120** — that's why we use transformers Whisper instead
- Models are loaded/unloaded sequentially to minimize peak VRAM
