#!/usr/bin/env bash
# DubbingStudio Start Script
# Starts Redis, Celery worker, and the FastAPI server.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       DubbingStudio Starting             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

FRONTEND_DIR="$ROOT_DIR/frontend"

cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

source venv/bin/activate

# Ensure data directories exist
mkdir -p ../data/{uploads,outputs,temp,models}

# Build frontend if not already built
if [ ! -d "$FRONTEND_DIR/dist" ]; then
    echo -e "${YELLOW}Building frontend...${NC}"
    cd "$FRONTEND_DIR"
    if command -v npm &>/dev/null; then
        npm install --silent 2>/dev/null
        npm run build 2>&1
        if [ -d "dist" ]; then
            echo -e "${GREEN}Frontend built${NC}"
        else
            echo -e "${YELLOW}WARNING: Frontend build failed — API will run without UI${NC}"
        fi
    else
        echo -e "${YELLOW}WARNING: npm not found — API will run without UI${NC}"
    fi
    cd "$BACKEND_DIR"
else
    echo -e "${GREEN}Frontend already built${NC}"
fi

# Start Redis if not running
if ! redis-cli ping &>/dev/null 2>&1; then
    echo -e "${YELLOW}Starting Redis...${NC}"
    redis-server --daemonize yes --logfile /tmp/dubbing-redis.log
    sleep 1
    if redis-cli ping &>/dev/null 2>&1; then
        echo -e "${GREEN}Redis started${NC}"
    else
        echo "ERROR: Failed to start Redis"
        exit 1
    fi
else
    echo -e "${GREEN}Redis already running${NC}"
fi

# Start Celery worker supervisor in background.
# Uses prefork pool with concurrency=1 and max-tasks-per-child=1 so that
# EVERY dubbing job runs in a freshly-spawned child process. When the job
# finishes the child exits and the OS reclaims all GPU/CUDA state —
# critical on Blackwell GPUs where Whisper/pyannote/NLLB leave CUDA
# contexts that can corrupt subsequent jobs.
#
# The supervisor loop restarts the master worker if it ever dies.
echo -e "${YELLOW}Starting Celery worker (supervised)...${NC}"

# Kill any stale supervisor / worker first
if [ -f /tmp/dubbing-celery-supervisor.pid ]; then
    OLD_SUP=$(cat /tmp/dubbing-celery-supervisor.pid 2>/dev/null || true)
    if [ -n "$OLD_SUP" ] && kill -0 "$OLD_SUP" 2>/dev/null; then
        kill "$OLD_SUP" 2>/dev/null || true
        sleep 1
    fi
fi
pkill -9 -f "celery.*-A app.core.celery_app.*worker" 2>/dev/null || true

(
    while true; do
        echo "[supervisor] starting celery worker at $(date)" >> /tmp/dubbing-celery-supervisor.log
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        PYTHONPATH="$ROOT_DIR/fish-speech:${PYTHONPATH:-}" \
        celery -A app.core.celery_app worker \
            -Q dubbing \
            --pool=prefork \
            --concurrency=1 \
            --max-tasks-per-child=1 \
            -l info \
            --logfile=/tmp/dubbing-celery.log \
            --pidfile=/tmp/dubbing-celery.pid \
            >> /tmp/dubbing-celery-supervisor.log 2>&1
        RC=$?
        echo "[supervisor] celery exited rc=$RC at $(date), restarting in 2s" >> /tmp/dubbing-celery-supervisor.log
        rm -f /tmp/dubbing-celery.pid
        sleep 2
    done
) &
SUPERVISOR_PID=$!
echo $SUPERVISOR_PID > /tmp/dubbing-celery-supervisor.pid
disown $SUPERVISOR_PID 2>/dev/null || true

sleep 3
echo -e "${GREEN}Celery worker started (supervisor pid=$SUPERVISOR_PID)${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   DubbingStudio ready at                 ║${NC}"
echo -e "${GREEN}║   http://localhost:8000                  ║${NC}"
echo -e "${GREEN}║   API docs: http://localhost:8000/docs   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Logs:"
echo "  Celery: tail -f /tmp/dubbing-celery.log"
echo "  Redis:  tail -f /tmp/dubbing-redis.log"
echo ""
echo "Press Ctrl+C to stop the server (Redis and Celery will keep running)"
echo "Run ./scripts/stop.sh to stop everything"
echo ""

# Start FastAPI server (foreground)
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --loop uvloop \
    --log-level info
