#!/usr/bin/env bash
# Start the frontend dev server with hot reload.
# Proxies API requests to the FastAPI backend on port 8000.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")/frontend"

cd "$FRONTEND_DIR"
npm run dev
