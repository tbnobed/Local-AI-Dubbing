# DubbingStudio

AI-powered video transcription, translation, and voice-cloned dubbing — runs entirely locally on your GPU.

## Features

- **Speech Transcription** — Whisper large-v3 with word-level timestamps (CUDA-accelerated)
- **Multi-Speaker Diarization** — Identifies and separates individual speakers via pyannote.audio
- **Translation** — NLLB-200 by Meta AI, supports 200+ languages (no cloud API needed)
- **Voice Cloning** — XTTS v2 by Coqui: each speaker's voice is cloned and used to synthesize translated speech
- **Audio Time-Stretch** — Synthesized segments are time-stretched to match original timing
- **SRT Subtitles** — Both original-language and translated SRT files exported automatically
- **Web UI** — Clean browser-based interface with real-time progress tracking
- **Multi-GPU** — Routes workloads across RTX 5090 (primary) and RTX 4500 (secondary)

## Supported Languages (expandable)

English, Spanish, French, German, Italian, Portuguese, Japanese, Chinese, Korean, Arabic, Russian, Hindi — and 190+ more via NLLB-200.

---

## Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended) or Windows 11 with WSL2
- **GPU**: NVIDIA RTX with CUDA 12.x (RTX 5090 + RTX 4500 Blackwell)
- **VRAM**: 16GB+ recommended (large-v3 Whisper + XTTS v2 simultaneously)
- **Python**: 3.10 or 3.11
- **Node.js**: 18+
- **ffmpeg**: installed and on PATH
- **Redis**: 7.x

---

## Quick Start

### 1. Clone / copy this directory to your local machine

### 2. Get a free Hugging Face token

Required for speaker diarization model:
1. Create account at https://huggingface.co
2. Get token at https://huggingface.co/settings/tokens
3. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Run setup

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 4. Configure

```bash
cd backend
cp .env.example .env
# Edit .env and set HF_TOKEN=your_token_here
```

### 5. Start everything

```bash
./scripts/start.sh
```

Open http://localhost:8000 in your browser.

---

## Architecture

```
Browser → FastAPI (port 8000)
                ↓
           Redis queue
                ↓
         Celery Worker
                ↓
    ┌──────────────────────┐
    │  Pipeline Stages:    │
    │  1. Audio extract    │ ← ffmpeg
    │  2. Transcribe       │ ← faster-whisper (CUDA:0)
    │  3. Diarize          │ ← pyannote (CUDA:1)
    │  4. Translate        │ ← NLLB-200 (CUDA:0)
    │  5. Voice clone TTS  │ ← XTTS v2 (CUDA:0)
    │  6. Mix & export     │ ← ffmpeg + librosa
    └──────────────────────┘
```

## Directory Structure

```
dubbing-studio/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes (jobs, websocket, system)
│   │   ├── core/         # Database, Celery, WebSocket manager
│   │   ├── models/       # SQLAlchemy Job model
│   │   ├── services/     # Transcription, diarization, translation, TTS, mixing
│   │   └── workers/      # Celery pipeline task
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── components/   # UploadForm, JobCard, SystemStatusBar
│       ├── hooks/        # useJobs, useJobWebSocket
│       ├── lib/          # API client
│       └── types/        # TypeScript interfaces
├── scripts/
│   ├── setup.sh          # One-time installation
│   ├── start.sh          # Start all services
│   ├── stop.sh           # Stop all services
│   ├── worker.sh         # Foreground worker (for debugging)
│   ├── build-frontend.sh # Rebuild React app
│   └── dev-frontend.sh   # Frontend dev server with hot reload
└── data/                 # Auto-created: uploads, outputs, temp, models
```

## GPU Configuration

Edit `backend/.env` to assign GPU IDs:

```env
PRIMARY_GPU_ID=0    # RTX 5090 — Whisper + Translation + XTTS
SECONDARY_GPU_ID=1  # RTX 4500 — Diarization
```

Run `nvidia-smi` to confirm GPU IDs on your system.

## Model Sizes

Whisper model sizes (trade-off speed vs. accuracy):
| Model | VRAM | Speed |
|-------|------|-------|
| large-v3 | ~10GB | Best accuracy (recommended) |
| medium | ~5GB | Good balance |
| small | ~2GB | Fastest |

NLLB translation models:
| Model | Size | VRAM |
|-------|------|------|
| nllb-200-distilled-600M | 600M | ~3GB (default) |
| nllb-200-1.3B | 1.3B | ~6GB |
| nllb-200-3.3B | 3.3B | ~13GB (best) |

## Troubleshooting

**Models download on first run** — Whisper (~3GB), NLLB (~2GB), XTTS v2 (~2GB), pyannote (~1GB). First job will be slow; subsequent jobs use cached models.

**Out of VRAM** — Switch to smaller models in `.env`, or increase `WHISPER_COMPUTE_TYPE=int8` for lower memory usage.

**pyannote 403 error** — You need to accept the model terms on Hugging Face and set `HF_TOKEN` in `.env`.

**Diarization disabled** — If `HF_TOKEN` is not set, the pipeline falls back to single-speaker mode (still works, just no per-speaker voice cloning).
