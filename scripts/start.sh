#!/usr/bin/env bash
# Bring up the full PKMS stack in order (watcher is a precondition):
#   1. Qdrant + Ollama (docker compose)   2. watcher: reconcile raw↔wiki → ready
#   3. web server (only after the watcher signals readiness)
#
# Usage:
#   scripts/start.sh          bring the stack up (foreground; Ctrl-C stops everything)
#   scripts/start.sh down     stop the web/watcher and bring the containers down
#
# NB: if you run Ollama natively on the host, stop it first so the container can
#     bind :11434 →  sudo systemctl stop ollama
set -euo pipefail
cd "$(dirname "$0")/.."

# ── down: tear everything down ────────────────────────────────────────────────
if [ "${1:-}" = "down" ]; then
  echo "[pkms] docker compose down…"
  docker compose down
  rm -f "$HOME/.pkms/watcher.ready"
  echo "[pkms] stack down."
  exit 0
fi

# Bounded wait: run a command until it succeeds, or fail after a timeout instead
# of hanging forever (a service that never comes up should abort, not block).
#   wait_for <timeout-seconds> <human-label> <cmd...>
wait_for() {
  local timeout="$1" label="$2"; shift 2
  local deadline=$(( SECONDS + timeout ))
  until "$@" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "[pkms] ERROR: timed out after ${timeout}s waiting for ${label}" >&2
      exit 1
    fi
    sleep 2
  done
}

echo "[pkms] starting Qdrant + Ollama (docker compose up -d)…"
docker compose up -d

echo "[pkms] waiting for Qdrant (:6333)…"
wait_for 60 "Qdrant :6333" curl -sf http://localhost:6333/healthz
echo "[pkms] waiting for Ollama (:11434)…"
wait_for 60 "Ollama :11434" curl -sf http://localhost:11434/api/tags

# /api/tags responds as soon as Ollama is up — NOT when ollama-init has finished
# pulling the models. Wait for the required models so the first ingest doesn't 404.
# Generous timeout: a cold pull of mistral:7b can take minutes.
echo "[pkms] waiting for models (ollama-init pulls mistral / nomic / llama3.2:1b)…"
models_ready() {
  local tags; tags=$(curl -sf http://localhost:11434/api/tags 2>/dev/null) || return 1
  grep -q 'mistral:7b' <<<"$tags" \
    && grep -q 'nomic-embed-text' <<<"$tags" \
    && grep -q 'llama3.2:1b' <<<"$tags"
}
wait_for 900 "Ollama models (mistral/nomic/llama3.2:1b)" models_ready
echo "[pkms] models ready ✓"

READY="$HOME/.pkms/watcher.ready"
rm -f "$READY"

echo "[pkms] starting watcher (reconcile raw↔wiki, then signal ready)…"
uv run python -c "import sys; sys.argv=['pkms','watch']; from pkms.coordinator import main; main()" &
WATCHER_PID=$!

cleanup() { echo "[pkms] stopping watcher…"; kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[pkms] waiting for watcher readiness (raw↔wiki consistent)…"
READY_DEADLINE=$(( SECONDS + 600 ))
until [ -f "$READY" ]; do
  kill -0 "$WATCHER_PID" 2>/dev/null || { echo "[pkms] watcher exited before ready — aborting" >&2; exit 1; }
  if [ "$SECONDS" -ge "$READY_DEADLINE" ]; then
    echo "[pkms] ERROR: watcher did not become ready within 600s — aborting" >&2
    exit 1
  fi
  sleep 1
done
echo "[pkms] watcher ready ✓"

echo "[pkms] starting web server on :8000 (Ctrl-C to stop everything)…"
uv run python -c "from pkms.web import serve; serve()"
