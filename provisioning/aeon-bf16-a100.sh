#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

readonly MODEL_ID="AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16"
readonly MODEL_REVISION="da9996c35307783ecaf25bdd240aeade954dd2c6"
readonly SERVED_MODEL_NAME="aeon-qwen3.6-27b-bf16"
readonly LAB_ROOT="/workspace/vast-text-inference"
readonly MODEL_DIR="${LAB_ROOT}/models/aeon-qwen3.6-27b-bf16"
readonly STATE_DIR="${LAB_ROOT}/state"
readonly LOG_DIR="${LAB_ROOT}/logs"
readonly SERVER_LOG="${LOG_DIR}/vllm.log"
readonly STATUS_FILE="${STATE_DIR}/status"
readonly PID_FILE="${STATE_DIR}/vllm.pid"

mkdir -p "${MODEL_DIR}" "${STATE_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/provisioning.log") 2>&1

mark_failed() {
  local exit_code=$?
  printf 'failed\n' >"${STATUS_FILE}"
  touch "${STATE_DIR}/failed"
  printf 'Provisioning failed with exit code %s.\n' "${exit_code}"
  exit "${exit_code}"
}
trap mark_failed ERR

if [[ -f "${STATE_DIR}/ready" ]] && python3 - <<'PY'
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
then
  printf 'ready\n' >"${STATUS_FILE}"
  printf 'vLLM is already healthy; nothing to do.\n'
  exit 0
fi

rm -f "${STATE_DIR}/ready" "${STATE_DIR}/failed"
printf 'downloading\n' >"${STATUS_FILE}"
printf 'Downloading %s at immutable revision %s...\n' "${MODEL_ID}" "${MODEL_REVISION}"

export HF_HOME="${LAB_ROOT}/huggingface"
export TOKENIZERS_PARALLELISM=false

python3 - "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision, model_dir = sys.argv[1:]
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=model_dir,
)
PY

printf 'starting\n' >"${STATUS_FILE}"
printf 'Starting conservative single-GPU BF16 baseline...\n'

nohup vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.86 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --attention-backend flash_attn \
  --mamba-cache-dtype float32 \
  --generation-config vllm \
  --trust-remote-code \
  >"${SERVER_LOG}" 2>&1 &

server_pid=$!
printf '%s\n' "${server_pid}" >"${PID_FILE}"

for _ in $(seq 1 210); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    printf 'vLLM exited during startup. Last log lines:\n'
    tail -n 120 "${SERVER_LOG}" || true
    false
  fi

  if python3 - <<'PY'
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except (urllib.error.URLError, TimeoutError):
    raise SystemExit(1)
PY
  then
    printf 'ready\n' >"${STATUS_FILE}"
    touch "${STATE_DIR}/ready"
    printf 'vLLM is healthy on remote loopback port 8000.\n'
    exit 0
  fi

  sleep 10
done

printf 'vLLM did not become healthy within 35 minutes.\n'
tail -n 120 "${SERVER_LOG}" || true
false
