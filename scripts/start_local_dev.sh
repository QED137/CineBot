#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found at $ROOT_DIR/.venv"
  exit 1
fi

echo "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
source .venv/bin/activate
python -m uvicorn app_fastapi:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

cleanup() {
  echo "Stopping local dev processes..."
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait briefly for backend startup
sleep 3

echo "Starting frontend on http://localhost:${FRONTEND_PORT}"
cd "$ROOT_DIR/frontend"
VITE_API_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/api" npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo ""
echo "Local dev is running"
echo "Frontend: http://localhost:${FRONTEND_PORT}"
echo "Backend health: http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
echo ""
wait
