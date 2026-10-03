#!/usr/bin/env bash
# ==============================================================================
# SmartLogi Agentic RAG Chatbot - Application Startup Script
# Thesis Demonstration Entry Point
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "======================================================================"
echo " Starting SmartLogi Agentic RAG Chatbot Server..."
echo "======================================================================"

# 1. Environment File Check
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[WARNING] .env file not found. Copying from .env.example..."
        cp .env.example .env
    else
        echo "[ERROR] Missing .env and .env.example files. Aborting startup."
        exit 1
    fi
fi

# Load environment variables
export $(grep -v '^#' .env | xargs) 2>/dev/null || true

# 2. Configuration Inspection & Validation
HOST="${FASTAPI_HOST:-0.0.0.0}"
PORT="${FASTAPI_PORT:-8000}"
DEMO="${DEMO_MODE:-true}"
WRITE_MODE="${LARK_WRITE_MODE:-mock}"

echo "[INFO] FastAPI Host:          $HOST"
echo "[INFO] FastAPI Port:          $PORT"
echo "[INFO] Demo Mode (DEMO_MODE): $DEMO"
echo "[INFO] Write Mode:            $WRITE_MODE"

if [ "$DEMO" = "true" ] || [ "$WRITE_MODE" = "mock" ]; [
    echo "[INFO] Running in SAFE DEMO MODE. Lark priority writes are simulated."
else
    if [ -z "$LARK_APP_ID" ] || [ -z "$LARK_APP_SECRET" ]; then
        echo "[ERROR] Real Lark write mode requested, but LARK_APP_ID or LARK_APP_SECRET is unconfigured."
        exit 1
    fi
    echo "[WARNING] Running in LIVE WRITE MODE against Lark Base application token."
fi

# 3. Start FastAPI Server via uvicorn
echo "======================================================================"
echo "[STATUS] Launching uvicorn server on http://$HOST:$PORT ..."
echo "======================================================================"

exec uvicorn src.api.api:app --host "$HOST" --port "$PORT" --reload
