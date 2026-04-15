# DubbingStudio

AI-powered video transcription, translation, and voice-cloned dubbing — runs entirely locally on your GPU.

## AI Model Stack (2025 State-of-the-Art)

| Stage | Model | Why This One |
|---|---|---|
| **Transcription** | **WhisperX** (large-v3-turbo) | 60x real-time, word-level timestamps, built-in VAD |
| **Speaker Diarization** | **pyannote community-1** (via WhisperX) | Best open-source diarization accuracy, exclusive alignment mode |
| **Translation** | **MADLAD-400 3B** (Google) | Apache 2.0 license, 400+ languages, better quality than NLLB |
| **Voice Cloning TTS** | **Fish Speech 1.5** | SOTA zero-shot voice cloning, 80+ languages, 15x real-time on RTX 4090 |
| **Audio Processing** | **ffmpeg + librosa** | Time-stretching, mixing, video assembly |

## Features

- **Speech Transcription** — WhisperX with word-level timestamps and VAD (CUDA-accelerated)
- **Multi-Speaker Diarization** — Identifies and separates individual speakers automatically
- **Translation** — MADLAD-400 by Google, 400+ languages, commercially usable (Apache 2.0)
- **Voice Cloning** — Fish Speech 1.5: each speaker's voice is cloned and used to synthesize translated speech
- **Audio Time-Stretch** — Synthesized segments are stretched to match original timing
- **SRT Subtitles** — Both original-language and translated SRT files exported automatically
- **Web UI** — Clean browser-based interface with real-time progress tracking via WebSocket
- **Multi-GPU** — Routes workloads across RTX 5090 (primary) and RTX 4500 (secondary)

## Supported Languages (expandable)

English, Spanish, French, German, Italian, Portuguese, Japanese, Chinese, Korean, Arabic, Russian, Hindi — and 390+ more via MADLAD-400.

---

## Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended) or Windows 11 with WSL2
- **GPU**: NVIDIA RTX with CUDA 12.x (RTX 5090 + RTX 4500 Blackwell)
- **VRAM**: 16GB+ per GPU recommended
- **Python**: 3.10, 3.11, or 3.12 (NOT 3.13 — WhisperX incompatible)
- **Node.js**: 18+
- **ffmpeg**: installed and on PATH
- **Redis**: 7.x

---

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Local-AI-Dubbing/dubbing-studio
```

### 2. Get a free Hugging Face token

Required for speaker diarization:
1. Create account at https://huggingface.co
2. Get token at https://huggingface.co/settings/tokens
3. Accept terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

### 3. Run setup

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 4. Configure

```bash
cd backend
# Edit .env and set HF_TOKEN=hf_your_token_here
nano .env
```

### 5. Install Redis (if not already installed)

```bash
sudo apt install redis-server
```

### 6. Start everything

```bash
./scripts/start.sh
```

Open http://localhost:8000 in your browser.

---

## Architecture

```
Browser (React UI)
    ↓
FastAPI Server (port 8000)
    ↓ upload + job creation
Redis Queue
    ↓
Celery Worker (GPU pipeline)
    ↓
┌────────────────────────────────────────────┐
│  1. Audio Extract           → ffmpeg       │
│  2. Transcribe + Align      → WhisperX     │  ← GPU:0
│  3. Speaker Diarization     → pyannote     │  ← GPU:0
│  4. Translation             → MADLAD-400   │  ← GPU:0
│  5. Voice Clone + TTS       → Fish Speech  │  ← GPU:0
│  6. Time-Stretch + Mix      → librosa      │
│  7. Final Video Assembly    → ffmpeg       │
└────────────────────────────────────────────┘
    ↓
Output: dubbed video + SRT files
```

## Directory Structure

```
dubbing-studio/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes (jobs, websocket, system status)
│   │   ├── core/         # Database, Celery, WebSocket manager
│   │   ├── models/       # SQLAlchemy Job model
│   │   ├── services/     # Transcription, translation, TTS, mixing
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
PRIMARY_GPU_ID=0    # RTX 5090 — WhisperX + MADLAD-400 + Fish Speech
SECONDARY_GPU_ID=1  # RTX 4500 — available for parallel workloads
```

Run `nvidia-smi` to confirm GPU IDs on your system.

## Model Sizes & VRAM

| Model | Download Size | VRAM (fp16) |
|---|---|---|
| WhisperX large-v3-turbo | ~3 GB | ~6 GB |
| MADLAD-400 3B | ~6 GB | ~7 GB |
| Fish Speech 1.5 | ~3 GB | ~5 GB |
| pyannote community-1 | ~200 MB | ~2 GB |

Models are loaded/unloaded between pipeline stages to minimize peak VRAM usage.

### Alternative model sizes

| Whisper Model | VRAM | Speed | Accuracy |
|---|---|---|---|
| large-v3-turbo | ~6GB | Fastest (recommended) | Best balance |
| large-v3 | ~10GB | Slower | Highest accuracy |
| medium | ~5GB | Fast | Good |
| small | ~2GB | Very fast | Acceptable |

| MADLAD Translation | VRAM | Quality |
|---|---|---|
| google/madlad400-3b-mt | ~7GB | Good (default) |
| google/madlad400-7b-mt | ~14GB | Better |
| google/madlad400-10b-mt | ~20GB | Best |

## Troubleshooting

**Models download on first run** — WhisperX (~3GB), MADLAD-400 (~6GB), Fish Speech (~3GB), pyannote (~200MB). First job will be slow; subsequent jobs use cached models.

**Out of VRAM** — Switch to smaller models in `.env`. You can use `WHISPER_COMPUTE_TYPE=int8` to halve Whisper memory. Or use `google/madlad400-3b-mt` instead of 7B/10B.

**pyannote 403 error** — You need to accept the model terms on HuggingFace AND set `HF_TOKEN` in `.env`.

**Diarization disabled** — If `HF_TOKEN` is not set, the pipeline falls back to single-speaker mode (still works, just no per-speaker voice cloning).

**WhisperX overwritten PyTorch** — If `torch.cuda.is_available()` returns False after setup, reinstall PyTorch: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`

**Fish Speech not found** — If pip install fails, install from source:
```bash
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech && pip install -e .[cu124]
```
