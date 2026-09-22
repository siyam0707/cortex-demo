#!/usr/bin/env bash
set -euo pipefail

# Required: local path to a downloaded/exported model. Model weights are never
# included in this repository.
: "${MODEL_PATH:?Set MODEL_PATH to the local Keural-Cortex-8B-v2 export directory.}"

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-vllm/bin/python}"
MODEL_NAME="${SERVED_MODEL_NAME:-Keural-cortex-v2}"
VLLM_PORT="${VLLM_PORT:-8000}"
ADAPTER_PORT="${ADAPTER_PORT:-8081}"
PUBLIC_TUNNEL="${PUBLIC_TUNNEL:-0}"

[[ -x "$VLLM_BIN" ]] || { echo "vLLM binary not found: $VLLM_BIN" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python binary not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Model directory not found: $MODEL_PATH" >&2; exit 1; }

cleanup() {
  [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "${ADAPTER_PID:-}" ]] && kill "$ADAPTER_PID" 2>/dev/null || true
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$VLLM_BIN" serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
  --chat-template "$MODEL_PATH/chat_template.jinja" --host 127.0.0.1 --port "$VLLM_PORT" \
  --max-model-len "${MAX_MODEL_LEN:-65536}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" --enforce-eager &
VLLM_PID=$!

for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM stopped; see terminal output." >&2; exit 1; }
  sleep 1
done
curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null || { echo "vLLM did not become ready." >&2; exit 1; }

PYTHONNOUSERSITE=1 VLLM_UPSTREAM="http://127.0.0.1:$VLLM_PORT" SERVED_MODEL_NAME="$MODEL_NAME" \
  "$PYTHON_BIN" -m uvicorn app:app --app-dir "$ROOT" --host 127.0.0.1 --port "$ADAPTER_PORT" --no-access-log &
ADAPTER_PID=$!
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:$ADAPTER_PORT/health" >/dev/null && break
  kill -0 "$ADAPTER_PID" 2>/dev/null || { echo "Adapter stopped; see terminal output." >&2; exit 1; }
  sleep 1
done
curl -fsS "http://127.0.0.1:$ADAPTER_PORT/health" >/dev/null || { echo "Adapter did not become ready." >&2; exit 1; }

echo "Cortex demo ready: http://127.0.0.1:$ADAPTER_PORT/v1"
if [[ "$PUBLIC_TUNNEL" == "1" ]]; then
  command -v cloudflared >/dev/null || { echo "cloudflared is required for PUBLIC_TUNNEL=1" >&2; exit 1; }
  cloudflared tunnel --url "http://127.0.0.1:$ADAPTER_PORT" --no-autoupdate &
  TUNNEL_PID=$!
  echo "Cloudflare will print a temporary HTTPS URL. Append /v1 in Keural Gateway."
fi
wait "$VLLM_PID"
