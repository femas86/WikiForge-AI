#!/usr/bin/env bash
# Bring up the full PKMS stack in order (watcher is a precondition):
#   1. Qdrant + Ollama (docker compose)   2. watcher: reconcile raw↔wiki → ready
#   3. web server (only after the watcher signals readiness)
#
# NB: if you run Ollama natively on the host, stop it first so the container can
#     bind :11434 →  sudo systemctl stop ollama
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[pkms] starting Qdrant + Ollama (docker compose up -d)…"
docker compose up -d

echo "[pkms] waiting for Qdrant (:6333)…"
until curl -sf http://localhost:6333/healthz >/dev/null 2>&1; do sleep 1; done
echo "[pkms] waiting for Ollama (:11434)…"
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done

# /api/tags responds as soon as Ollama is up — NOT when ollama-init has finished
# pulling the models. Wait for the required models so the first ingest doesn't 404.
echo "[pkms] waiting for models (ollama-init pulls mistral / nomic / llama3.2:1b)…"
until tags=$(curl -sf http://localhost:11434/api/tags 2>/dev/null) \
      && grep -q 'mistral:7b' <<<"$tags" \
      && grep -q 'nomic-embed-text' <<<"$tags" \
      && grep -q 'llama3.2:1b' <<<"$tags"; do
  sleep 3
done
echo "[pkms] models ready ✓"

READY="$HOME/.pkms/watcher.ready"
rm -f "$READY"

echo "[pkms] starting watcher (reconcile raw↔wiki, then signal ready)…"
uv run python -c "import sys; sys.argv=['pkms','watch']; from pkms.coordinator import main; main()" &
WATCHER_PID=$!

cleanup() { echo "[pkms] stopping watcher…"; kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[pkms] waiting for watcher readiness (raw↔wiki consistent)…"
until [ -f "$READY" ]; do
  kill -0 "$WATCHER_PID" 2>/dev/null || { echo "[pkms] watcher exited before ready — aborting"; exit 1; }
  sleep 1
done
echo "[pkms] watcher ready ✓"

echo "[pkms] starting web server on :8000 (Ctrl-C to stop everything)…"
uv run python -c "from pkms.web import serve; serve()"
