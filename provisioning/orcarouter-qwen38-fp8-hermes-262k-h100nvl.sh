#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

readonly MODEL_ID="orcarouter/Qwen3.8-27B-Uncensored-FP8"
readonly MODEL_REVISION="0787858da83e6640e289c0c22d092d92f4e97fdb"
readonly SERVED_MODEL_NAME="qwen3.8-27b-uncensored-fp8-262k"
readonly LAB_ROOT="/workspace/vast-text-inference"
readonly MODEL_DIR="${LAB_ROOT}/models/qwen3.8-27b-uncensored-fp8"
readonly STATE_DIR="${LAB_ROOT}/state"
readonly LOG_DIR="${LAB_ROOT}/logs"
readonly VLLM_LOG="${LOG_DIR}/vllm.log"
readonly STATUS_FILE="${STATE_DIR}/status"
readonly VLLM_PID_FILE="${STATE_DIR}/vllm.pid"

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

http_ok() {
  local url=$1
  python3 - "${url}" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except (urllib.error.URLError, TimeoutError):
    raise SystemExit(1)
PY
}

api_smoke_ok() {
  python3 - "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request

model = sys.argv[1]
url = "http://127.0.0.1:8000/v1/chat/completions"

def post(payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.load(response)

chat = post({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: HERMES_262K_OK"}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
})
content = chat["choices"][0]["message"].get("content") or ""
if "HERMES_262K_OK" not in content:
    raise RuntimeError("chat smoke test did not return the marker")

tool = post({
    "model": model,
    "messages": [{"role": "user", "content": "What is the weather in Brussels?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
})
tool_calls = tool["choices"][0]["message"].get("tool_calls") or []
if not tool_calls or tool_calls[0].get("function", {}).get("name") != "get_weather":
    raise RuntimeError("forced tool-call smoke test failed")
PY
}

if [[ -f "${STATE_DIR}/ready" ]] \
  && http_ok "http://127.0.0.1:8000/health" \
  && api_smoke_ok; then
  printf 'ready\n' >"${STATUS_FILE}"
  printf 'The Hermes 262K endpoint is already healthy; nothing to do.\n'
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  printf 'HF_TOKEN is required because the pinned model repository is gated.\n' >&2
  false
fi

rm -f "${STATE_DIR}/ready" "${STATE_DIR}/failed"
printf 'downloading\n' >"${STATUS_FILE}"
printf 'Downloading %s at immutable revision %s...\n' "${MODEL_ID}" "${MODEL_REVISION}"

export HF_HOME="${LAB_ROOT}/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

python3 - "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_DIR}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model_id, revision, model_dir = sys.argv[1:]
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=model_dir,
    token=os.environ["HF_TOKEN"],
)
PY

printf 'starting-vllm\n' >"${STATUS_FILE}"
printf 'Starting Qwen3.8 FP8 for Hermes with the full 262144-token context...\n'

nohup vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port 8000 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --trust-remote-code \
  >"${VLLM_LOG}" 2>&1 &

vllm_pid=$!
printf '%s\n' "${vllm_pid}" >"${VLLM_PID_FILE}"

for _ in $(seq 1 360); do
  if ! kill -0 "${vllm_pid}" 2>/dev/null; then
    printf 'vLLM exited during startup. Last log lines:\n'
    tail -n 120 "${VLLM_LOG}" || true
    false
  fi

  if http_ok "http://127.0.0.1:8000/health"; then
    break
  fi
  sleep 10
done

if ! http_ok "http://127.0.0.1:8000/health"; then
  printf 'vLLM did not become healthy within 60 minutes.\n'
  tail -n 120 "${VLLM_LOG}" || true
  false
fi

printf 'testing-hermes-api\n' >"${STATUS_FILE}"
if ! api_smoke_ok; then
  printf 'Hermes chat or forced tool-call smoke test failed.\n'
  tail -n 120 "${VLLM_LOG}" || true
  false
fi

printf 'ready\n' >"${STATUS_FILE}"
touch "${STATE_DIR}/ready"
printf 'Hermes 262K chat and forced tool-call endpoint is healthy.\n'
