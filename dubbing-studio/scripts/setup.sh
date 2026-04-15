#!/usr/bin/env bash
# DubbingStudio Setup Script
# Run this once on your local machine to install all dependencies.
# Updated for: WhisperX + MADLAD-400 + Fish Speech 1.5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
FISH_SPEECH_DIR="$ROOT_DIR/fish-speech"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       DubbingStudio Setup                ║"
echo "║  WhisperX + MADLAD-400 + Fish Speech     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Python: $python_version"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) and sys.version_info < (3, 13) else 1)" 2>/dev/null; then
    echo "Python version OK (3.10-3.12 required)"
else
    echo "ERROR: Python 3.10, 3.11, or 3.12 required."
    echo "Python 3.13 is NOT supported by WhisperX."
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. CUDA support may not be available."
fi

# Detect CUDA version for PyTorch index URL
CUDA_INDEX="cu124"
if command -v nvidia-smi &>/dev/null; then
    CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "Driver version: $CUDA_DRIVER_VER"

    # Check if we have a Blackwell GPU (RTX 50xx, sm_120) needing CUDA 12.8+
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if echo "$GPU_NAME" | grep -qiE "RTX 50[0-9]{2}|Blackwell"; then
        echo "Detected Blackwell-architecture GPU: $GPU_NAME"
        echo "Using CUDA 12.8 index for sm_120 support."
        CUDA_INDEX="cu128"
    fi
fi
echo "PyTorch CUDA index: $CUDA_INDEX"

# Check / install ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found — required for audio/video processing."
    if command -v apt &>/dev/null; then
        echo "Installing ffmpeg via apt..."
        sudo apt install -y ffmpeg
    elif command -v brew &>/dev/null; then
        echo "Installing ffmpeg via Homebrew..."
        brew install ffmpeg
    else
        echo "ERROR: Cannot auto-install ffmpeg. Please install manually."
        exit 1
    fi
fi
echo "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# Check / install Redis
if ! command -v redis-server &>/dev/null; then
    echo "Redis not found — required for job queuing."
    if command -v apt &>/dev/null; then
        echo "Installing Redis via apt..."
        sudo apt install -y redis-server
    elif command -v brew &>/dev/null; then
        echo "Installing Redis via Homebrew..."
        brew install redis
    else
        echo "ERROR: Cannot auto-install Redis. Please install manually."
        exit 1
    fi
fi
echo "Redis: $(redis-server --version 2>&1 | head -1)"

# Check / install Node.js (required for frontend build)
if ! command -v npm &>/dev/null; then
    echo "Node.js/npm not found — required for frontend build."
    if command -v apt &>/dev/null; then
        echo "Installing Node.js 20.x via NodeSource..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt install -y nodejs
    elif command -v brew &>/dev/null; then
        echo "Installing Node.js via Homebrew..."
        brew install node
    else
        echo "ERROR: Cannot auto-install Node.js. Please install manually from https://nodejs.org"
        exit 1
    fi
fi
echo "Node.js: $(node --version 2>&1)"
echo "npm: $(npm --version 2>&1)"

echo ""
echo "──────────────────────────────────────────"
echo "Step 1: Python virtual environment"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"

# Recreate venv if it exists but has stale/conflicting packages
if [ -d "venv" ]; then
    echo "Removing old virtual environment to get a clean slate..."
    rm -rf venv
fi

python3 -m venv venv
echo "Created virtual environment: $BACKEND_DIR/venv"

source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

echo ""
echo "──────────────────────────────────────────"
echo "Step 2: PyTorch + WhisperX + onnxruntime"
echo "──────────────────────────────────────────"

# Install WhisperX first — it pulls in faster-whisper, pyannote, etc.
# It may install CPU-only PyTorch, which we fix immediately after.
echo "Installing WhisperX (transcription + diarization)..."
pip install whisperx -q

# Re-pin PyTorch with CUDA (WhisperX may have pulled CPU-only builds)
echo "Pinning PyTorch with CUDA ($CUDA_INDEX)..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" -q

# onnxruntime-gpu replaces onnxruntime for GPU-accelerated diarization.
# faster-whisper requires onnxruntime>=1.14 — onnxruntime-gpu satisfies this.
echo "Installing onnxruntime-gpu..."
pip install onnxruntime-gpu -q 2>/dev/null || {
    echo "onnxruntime-gpu install failed, keeping onnxruntime (CPU diarization)"
}

echo ""
echo "──────────────────────────────────────────"
echo "Step 3: Core backend dependencies"
echo "──────────────────────────────────────────"

pip install -r requirements.txt -q

echo ""
echo "──────────────────────────────────────────"
echo "Step 4: Fish Speech (voice cloning TTS)"
echo "──────────────────────────────────────────"

cd "$ROOT_DIR"
if [ ! -d "$FISH_SPEECH_DIR" ]; then
    echo "Cloning Fish Speech repository..."
    git clone https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_DIR"
else
    echo "Fish Speech repository already exists, pulling latest..."
    cd "$FISH_SPEECH_DIR" && git pull && cd "$ROOT_DIR"
fi

# Ensure venv is still active after cd
source "$BACKEND_DIR/venv/bin/activate"

cd "$FISH_SPEECH_DIR"
echo "Installing Fish Speech (pip install -e .[$CUDA_INDEX])..."
pip install -e ".[$CUDA_INDEX]" -q 2>/dev/null || {
    echo "Fish Speech editable install with [$CUDA_INDEX] failed, trying default..."
    pip install -e . -q 2>/dev/null || {
        echo ""
        echo "WARNING: Fish Speech install failed."
        echo "You may need to install it manually:"
        echo "  cd $FISH_SPEECH_DIR"
        echo "  pip install -e .[$CUDA_INDEX]"
        echo ""
    }
}

# Re-pin PyTorch CUDA one final time (fish-speech may also overwrite)
echo "Final PyTorch CUDA re-pin..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" -q

# Download model checkpoint
echo "Downloading Fish Speech 1.5 model weights (~3GB)..."
cd "$BACKEND_DIR"
mkdir -p data/models/fish-speech
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading fishaudio/fish-speech-1.5...')
path = snapshot_download(
    'fishaudio/fish-speech-1.5',
    local_dir='data/models/fish-speech/fish-speech-1.5',
)
print(f'Downloaded to: {path}')
" || echo "Model download deferred — will download on first run."

echo ""
echo "──────────────────────────────────────────"
echo "Step 5: Verify GPU setup"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"
source "$BACKEND_DIR/venv/bin/activate"

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'  GPU {i}: {name} ({mem:.0f} GB)')
else:
    print('WARNING: CUDA not detected — pipeline will be very slow on CPU')

try:
    import whisperx
    print(f'WhisperX: OK')
except ImportError as e:
    print(f'WhisperX: MISSING ({e})')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers: MISSING ({e})')

try:
    import fish_speech
    print(f'Fish Speech: OK')
except ImportError:
    try:
        from fish_speech.inference_engine import TTSInferenceEngine
        print(f'Fish Speech (engine): OK')
    except ImportError as e:
        print(f'Fish Speech: MISSING ({e})')
" || echo "Could not verify full setup"

echo ""
echo "──────────────────────────────────────────"
echo "Step 6: Frontend"
echo "──────────────────────────────────────────"

cd "$FRONTEND_DIR"
if command -v npm &>/dev/null; then
    npm install
    echo "Building frontend..."
    npm run build
    echo "Frontend built successfully."
else
    echo "WARNING: npm not found. Install Node.js from https://nodejs.org"
fi

echo ""
echo "──────────────────────────────────────────"
echo "Step 7: Configuration"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  ACTION REQUIRED: Set your Hugging Face token           ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  1. Get a free token at:                                ║"
    echo "║     https://huggingface.co/settings/tokens              ║"
    echo "║                                                         ║"
    echo "║  2. Accept model terms at:                              ║"
    echo "║     https://huggingface.co/pyannote/speaker-diarization-3.1  ║"
    echo "║     https://huggingface.co/pyannote/segmentation-3.0    ║"
    echo "║                                                         ║"
    echo "║  3. Edit $BACKEND_DIR/.env                              ║"
    echo "║     Set HF_TOKEN=hf_your_token_here                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
else
    echo ".env already exists (not overwriting)"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  Models used:"
echo "    Transcription:  WhisperX (large-v3-turbo)"
echo "    Diarization:    pyannote community-1 (via WhisperX)"
echo "    Translation:    MADLAD-400 3B (Google, Apache 2.0)"
echo "    Voice cloning:  Fish Speech 1.5"
echo ""
echo "  To start DubbingStudio:"
echo "    ./scripts/start.sh"
echo ""
echo "  Or start manually:"
echo "    Terminal 1: redis-server"
echo "    Terminal 2: cd backend && source venv/bin/activate && celery -A app.core.celery_app worker -Q dubbing --concurrency=1 -l info"
echo "    Terminal 3: cd backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  Then open: http://localhost:8000"
echo ""
