#!/usr/bin/env bash
# Start the React + FastAPI application on a single port.
# This script:
#   1. Builds the React frontend (if not already built)
#   2. Starts the FastAPI backend which serves both API and static files

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_DIR="$SCRIPT_DIR/backend"

# Build frontend if dist/ doesn't exist
if [ ! -d "$FRONTEND_DIR/dist" ]; then
    echo "[Build] Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm install --production=false
    echo "[Build] Building React frontend..."
    npm run build
    cd "$SCRIPT_DIR"
fi

# Start FastAPI server
echo "[Start] Launching FastAPI server on port 7860..."
cd "$(dirname "$SCRIPT_DIR")"  # cd to Final-Web-Demo root
exec python -m uvicorn react_app.backend.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 1
