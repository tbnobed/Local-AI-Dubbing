#!/usr/bin/env bash
# Build the React frontend and output to backend's static dir.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")/frontend"

echo "Building frontend..."
cd "$FRONTEND_DIR"
npm run build
echo "Frontend built to: $FRONTEND_DIR/dist"
echo "The FastAPI server will serve it at http://localhost:8000"
