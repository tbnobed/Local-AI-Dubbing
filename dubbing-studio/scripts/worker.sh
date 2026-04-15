#!/usr/bin/env bash
# Start Celery worker in foreground with GPU logging.
# Useful for monitoring worker activity during processing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"

cd "$BACKEND_DIR"
source venv/bin/activate

# Configure GPU memory for multiple models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_DEVICE_ORDER=PCI_BUS_ID

celery -A app.core.celery_app worker \
    -Q dubbing \
    --concurrency=1 \
    -l info \
    --without-gossip \
    --without-mingle \
    --without-heartbeat
