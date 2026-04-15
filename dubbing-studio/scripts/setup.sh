#!/usr/bin/env bash
# DubbingStudio Setup Script
# Installs ALL system deps + Python packages + models + frontend.
# Safe to re-run — will clean and rebuild the Python environment each time.
#
# Stack: WhisperX + MADLAD-400 + Fish Speech 1.5
# Target: RTX 5090 (sm_120 Blackwell) + RTX 4500 Blackwell

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

# ─────────────────────────────────────────────────
# Pre-flight checks + system dependency installs
# ─────────────────────────────────────────────────

# Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Python: $python_version"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) and sys.version_info < (3, 13) else 1)" 2>/dev/null; then
    echo "  OK (3.10-3.12 required)"
else
    echo "  ERROR: Python 3.10, 3.11, or 3.12 required. (3.13 is NOT supported by WhisperX)"
    exit 1
fi

# GPU detection
CUDA_INDEX="cu124"
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "Driver: $CUDA_DRIVER_VER"

    # Blackwell (RTX 50xx / sm_120) needs CUDA 12.8+
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    if echo "$GPU_NAMES" | grep -qiE "RTX 50[0-9]{2}|Blackwell"; then
        echo "Detected Blackwell GPU — using CUDA 12.8 wheels."
        CUDA_INDEX="cu128"
    fi
else
    echo "WARNING: nvidia-smi not found. CUDA support may not be available."
fi
echo "PyTorch CUDA index: $CUDA_INDEX"
echo ""

# ── System packages (auto-install via apt/brew) ──

install_if_missing() {
    local cmd="$1" pkg_apt="$2" pkg_brew="$3" label="$4"
    if command -v "$cmd" &>/dev/null; then
        return 0
    fi
    echo "$label not found — installing..."
    if command -v apt &>/dev/null; then
        sudo apt install -y $pkg_apt
    elif command -v brew &>/dev/null; then
        brew install $pkg_brew
    else
        echo "ERROR: Cannot auto-install $label. Please install manually."
        exit 1
    fi
}

install_if_missing ffmpeg ffmpeg ffmpeg "ffmpeg"
echo "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

install_if_missing redis-server redis-server redis "Redis"
echo "Redis: $(redis-server --version 2>&1 | head -1)"

# portaudio dev headers — required by Fish Speech's pyaudio dependency
if command -v apt &>/dev/null; then
    if ! dpkg -s portaudio19-dev &>/dev/null 2>&1; then
        echo "Installing portaudio19-dev (needed by Fish Speech)..."
        sudo apt install -y portaudio19-dev
    fi
    echo "portaudio: installed"
elif command -v brew &>/dev/null; then
    brew list portaudio &>/dev/null 2>&1 || brew install portaudio
    echo "portaudio: installed"
fi

if ! command -v npm &>/dev/null; then
    echo "Node.js/npm not found — installing Node.js 20.x..."
    if command -v apt &>/dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt install -y nodejs
    elif command -v brew &>/dev/null; then
        brew install node
    else
        echo "ERROR: Cannot auto-install Node.js. Install from https://nodejs.org"
        exit 1
    fi
fi
echo "Node.js: $(node --version 2>&1)"
echo "npm: $(npm --version 2>&1)"

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 1: Python virtual environment"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"

# Always recreate for a clean dependency graph
if [ -d "venv" ]; then
    echo "Removing old venv for clean install..."
    rm -rf venv
fi

python3 -m venv venv
echo "Created venv: $BACKEND_DIR/venv"
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 2: PyTorch CUDA + WhisperX (alignment/diarization only)"
echo "──────────────────────────────────────────"

# WhisperX is used ONLY for wav2vec2 alignment + pyannote diarization.
# Transcription uses pure PyTorch Whisper (transformers) because
# CTranslate2/faster-whisper does NOT support Blackwell sm_120 GPUs.

# Install WhisperX normally first — it pulls its own deps
echo "  [2a] Installing WhisperX + all its dependencies..."
pip install whisperx -q

# Install WhisperX's missing/needed deps explicitly
echo "  [2b] Installing WhisperX supplementary deps..."
pip install nltk omegaconf pyannote.audio -q

# Re-pin PyTorch to CUDA build (WhisperX pulls CPU-only torch)
echo "  [2c] Re-pinning PyTorch 2.8.0 CUDA ($CUDA_INDEX)..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" --force-reinstall -q

# onnxruntime-gpu for GPU-accelerated speaker diarization
echo "  [2d] Installing onnxruntime-gpu..."
pip install onnxruntime-gpu -q 2>/dev/null || {
    echo "  onnxruntime-gpu failed — diarization will use CPU (still works)"
}

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 3: Backend dependencies"
echo "──────────────────────────────────────────"

pip install -r requirements.txt -q

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 4: Fish Speech (voice cloning TTS)"
echo "──────────────────────────────────────────"

cd "$ROOT_DIR"
if [ ! -d "$FISH_SPEECH_DIR" ]; then
    echo "  [4a] Cloning Fish Speech repo (v1.5.1)..."
    git clone --branch v1.5.1 --depth 1 https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_DIR"
else
    echo "  [4a] Fish Speech repo exists, ensuring v1.5.1 tag..."
    cd "$FISH_SPEECH_DIR"
    git fetch --tags 2>/dev/null || true
    git checkout v1.5.1 2>/dev/null || git checkout tags/v1.5.1 2>/dev/null || true
    cd "$ROOT_DIR"
fi

# Re-activate venv (cd may have broken it)
source "$BACKEND_DIR/venv/bin/activate"

cd "$FISH_SPEECH_DIR"
echo "  [4b] Installing Fish Speech (editable)..."

# Clear any corrupted pip wheel cache (pyaudio is a common offender)
pip cache remove pyaudio 2>/dev/null || true

# Try with CUDA extra first, then bare install, then no-build-isolation
FISH_INSTALLED=false
for attempt in ".[${CUDA_INDEX}]" "." ". --no-build-isolation"; do
    echo "    Trying: pip install -e $attempt ..."
    if pip install -e $attempt -q 2>&1; then
        FISH_INSTALLED=true
        echo "    Fish Speech installed successfully."
        break
    fi
done

if [ "$FISH_INSTALLED" = false ]; then
    echo ""
    echo "  WARNING: Fish Speech install failed."
    echo "  Try manually: cd $FISH_SPEECH_DIR && pip install -e ."
    echo ""
fi

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 5: Lock down final package versions"
echo "──────────────────────────────────────────"

# Fish Speech may have overwritten PyTorch.
# This final step guarantees the correct CUDA version is in place.

cd "$BACKEND_DIR"
source "$BACKEND_DIR/venv/bin/activate"

echo "  [5a] Final PyTorch 2.8.0 CUDA pin ($CUDA_INDEX)..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" --force-reinstall -q

echo "  [5b] Installing accelerate (required for transformers Whisper)..."
pip install "accelerate>=1.2.0" -q

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 6: Download model weights"
echo "──────────────────────────────────────────"

mkdir -p data/models/fish-speech
echo "  Downloading Fish Speech 1.5 (~3 GB)..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'fishaudio/fish-speech-1.5',
    local_dir='data/models/fish-speech/fish-speech-1.5',
)
print(f'  Downloaded to: {path}')
" || echo "  Deferred — will download on first run."

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 7: Verify GPU + packages"
echo "──────────────────────────────────────────"

python3 << 'VERIFY'
import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  {label}: {result}")
        return True
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
        return False

# PyTorch + CUDA
import torch
check("PyTorch", lambda: torch.__version__)
check("CUDA available", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    check("CUDA version", lambda: torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem = (props.total_memory if hasattr(props, 'total_memory') else props.total_mem) / 1024**3
        print(f"  GPU {i}: {name} (sm_{cap[0]}{cap[1]}, {mem:.0f} GB)")

# Transformers (used for Whisper transcription — pure PyTorch, Blackwell-safe)
check("Transformers", lambda: __import__('transformers').__version__)

# Accelerate (required by transformers for model loading)
check("Accelerate", lambda: __import__('accelerate').__version__)

# WhisperX (used for alignment + diarization only)
check("WhisperX", lambda: (__import__('whisperx'), "OK")[1])

# Fish Speech
try:
    __import__('fish_speech')
    print("  Fish Speech: OK")
except ImportError:
    print("  Fish Speech: not importable (may still work via CLI fallback)")

# CUDA smoke test — catches sm_120 kernel issues
if torch.cuda.is_available():
    try:
        x = torch.randn(16, 16, device='cuda:0')
        y = x @ x.T
        del x, y
        torch.cuda.empty_cache()
        print("  CUDA smoke test: PASSED")
    except Exception as e:
        print(f"  CUDA smoke test: FAILED — {e}")
        print("    Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall")
        sys.exit(1)
VERIFY

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 8: Frontend build"
echo "──────────────────────────────────────────"

cd "$FRONTEND_DIR"
echo "  Installing npm packages..."
npm install --silent 2>/dev/null
echo "  Building React app..."
npm run build
if [ -d "dist" ]; then
    echo "  Frontend built successfully."
else
    echo "  WARNING: Frontend build failed — API will run without UI."
fi

# ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "Step 9: Configuration"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo ""
    echo "  ╔═══════════════════════════════════════════════════════╗"
    echo "  ║  ACTION REQUIRED: Set your Hugging Face token        ║"
    echo "  ╠═══════════════════════════════════════════════════════╣"
    echo "  ║  1. Get a free token at:                             ║"
    echo "  ║     https://huggingface.co/settings/tokens           ║"
    echo "  ║                                                      ║"
    echo "  ║  2. Accept model terms at:                           ║"
    echo "  ║     huggingface.co/pyannote/speaker-diarization-3.1  ║"
    echo "  ║     huggingface.co/pyannote/segmentation-3.0         ║"
    echo "  ║                                                      ║"
    echo "  ║  3. Edit backend/.env → set HF_TOKEN=hf_xxxxx       ║"
    echo "  ╚═══════════════════════════════════════════════════════╝"
else
    echo "  .env already exists (not overwriting)"
fi

# ─────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  Models:"
echo "    Transcription:  WhisperX (large-v3-turbo)"
echo "    Diarization:    pyannote community-1 (via WhisperX)"
echo "    Translation:    MADLAD-400 3B (Google, Apache 2.0)"
echo "    Voice cloning:  Fish Speech 1.5"
echo ""
echo "  Next steps:"
echo "    1. Edit backend/.env and set HF_TOKEN (if not done)"
echo "    2. Run:  ./start.sh"
echo "    3. Open: http://localhost:8000"
echo ""
