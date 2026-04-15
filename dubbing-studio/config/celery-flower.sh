#!/usr/bin/env bash
# Optional: Start Flower - Celery task monitoring UI at http://localhost:5555

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")/backend"

cd "$BACKEND_DIR"
source venv/bin/activate
pip install flower -q
celery -A app.core.celery_app flower --port=5555
