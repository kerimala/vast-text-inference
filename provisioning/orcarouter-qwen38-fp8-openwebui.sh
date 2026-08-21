#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

readonly MODEL_ID="orcarouter/Qwen3.8-27B-Uncensored-FP8"
readonly MODEL_REVISION="0787858da83e6640e289c0c22d092d92f4e97fdb"
readonly SERVED_MODEL_NAME="qwen3.8-27b-uncensored-fp8"
readonly OPENWEBUI_VERSION="0.10.2"
readonly LAB_ROOT="/workspace/vast-text-inference"
readonly MODEL_DIR="${LAB_ROOT}/models/qwen3.8-27b-uncensored-fp8"
readonly OPENWEBUI_VENV="${LAB_ROOT}/openwebui-venv"
readonly OPENWEBUI_DATA_DIR="${LAB_ROOT}/openwebui-data"
readonly STATE_DIR="${LAB_ROOT}/state"
readonly LOG_DIR="${LAB_ROOT}/logs"
readonly VLLM_LOG="${LOG_DIR}/vllm.log"
readonly OPENWEBUI_LOG="${LOG_DIR}/openwebui.log"
readonly STATUS_FILE="${STATE_DIR}/status"
readonly VLLM_PID_FILE="${STATE_DIR}/vllm.pid"
readonly OPENWEBUI_PID_FILE="${STATE_DIR}/openwebui.pid"
readonly OPENWEBUI_WEB_SEARCH_ENGINE="duckduckgo"
readonly OPENWEBUI_WEB_SEARCH_RESULT_COUNT="5"
readonly OPENWEBUI_WEB_SEARCH_CONCURRENT_REQUESTS="2"

mkdir -p "${MODEL_DIR}" "${OPENWEBUI_DATA_DIR}" "${STATE_DIR}" "${LOG_DIR}"
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

web_search_ok() {
  "${OPENWEBUI_VENV}/bin/python" - <<'PY'
from ddgs import DDGS

results = DDGS().text("Open WebUI project", max_results=1)
if not results or not results[0].get("href"):
    raise SystemExit(1)
PY
}

if [[ -f "${STATE_DIR}/ready" ]] \
  && http_ok "http://127.0.0.1:8000/health" \
  && http_ok "http://127.0.0.1:3000/health"; then
  printf 'ready\n' >"${STATUS_FILE}"
  printf 'vLLM and Open WebUI are already healthy; nothing to do.\n'
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
printf 'Starting Qwen3.8 FP8 on remote loopback...\n'

nohup vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port 8000 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --trust-remote-code \
  >"${VLLM_LOG}" 2>&1 &

vllm_pid=$!
printf '%s\n' "${vllm_pid}" >"${VLLM_PID_FILE}"

for _ in $(seq 1 240); do
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
  printf 'vLLM did not become healthy within 40 minutes.\n'
  tail -n 120 "${VLLM_LOG}" || true
  false
fi

printf 'installing-openwebui\n' >"${STATUS_FILE}"
if [[ ! -x "${OPENWEBUI_VENV}/bin/open-webui" ]]; then
  python3 -m venv "${OPENWEBUI_VENV}"
  "${OPENWEBUI_VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    "open-webui==${OPENWEBUI_VERSION}"
fi

installed_openwebui_version=$("${OPENWEBUI_VENV}/bin/python" - <<'PY'
from importlib.metadata import version
print(version("open-webui"))
PY
)
if [[ "${installed_openwebui_version}" != "${OPENWEBUI_VERSION}" ]]; then
  printf 'Expected Open WebUI %s, found %s.\n' \
    "${OPENWEBUI_VERSION}" "${installed_openwebui_version}" >&2
  false
fi

printf 'starting-openwebui\n' >"${STATUS_FILE}"
webui_url="${WEBUI_PUBLIC_URL:-http://127.0.0.1:3000}"
cors_allow_origin="${webui_url};http://127.0.0.1:3000;http://localhost:3000"

(
  cd "${OPENWEBUI_DATA_DIR}"
  export DATA_DIR="${OPENWEBUI_DATA_DIR}"
  export WEBUI_URL="${webui_url}"
  export CORS_ALLOW_ORIGIN="${cors_allow_origin}"
  export OPENAI_API_BASE_URLS="http://127.0.0.1:8000/v1"
  export OPENAI_API_KEYS="local-vllm"
  export ENABLE_OLLAMA_API=false
  export DEFAULT_USER_ROLE=pending
  export ENABLE_WEB_SEARCH=true
  export ENABLE_WEB_SEARCH_CONFIRMATION=false
  export WEB_SEARCH_ENGINE="${OPENWEBUI_WEB_SEARCH_ENGINE}"
  export WEB_SEARCH_RESULT_COUNT="${OPENWEBUI_WEB_SEARCH_RESULT_COUNT}"
  export WEB_SEARCH_CONCURRENT_REQUESTS="${OPENWEBUI_WEB_SEARCH_CONCURRENT_REQUESTS}"
  export DDGS_BACKEND=auto
  export BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=false
  export BYPASS_WEB_SEARCH_WEB_LOADER=false
  export WEB_LOADER_CONCURRENT_REQUESTS=2
  export WEB_LOADER_TIMEOUT=20
  export ENABLE_WEB_LOADER_SSL_VERIFICATION=true
  export USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
  export DO_NOT_TRACK=true
  export SCARF_NO_ANALYTICS=true
  export ANONYMIZED_TELEMETRY=false
  export UVICORN_WORKERS=1
  if [[ "${webui_url}" == https://* ]]; then
    export WEBUI_SESSION_COOKIE_SECURE=true
    export WEBUI_AUTH_COOKIE_SECURE=true
  fi
  exec "${OPENWEBUI_VENV}/bin/open-webui" serve --host 127.0.0.1 --port 3000
) >"${OPENWEBUI_LOG}" 2>&1 &

openwebui_pid=$!
printf '%s\n' "${openwebui_pid}" >"${OPENWEBUI_PID_FILE}"

for _ in $(seq 1 120); do
  if ! kill -0 "${openwebui_pid}" 2>/dev/null; then
    printf 'Open WebUI exited during startup. Last log lines:\n'
    tail -n 120 "${OPENWEBUI_LOG}" || true
    false
  fi

  if http_ok "http://127.0.0.1:3000/health"; then
    break
  fi
  sleep 10
done

if ! http_ok "http://127.0.0.1:3000/health"; then
  printf 'Open WebUI did not become healthy within 20 minutes.\n'
  tail -n 120 "${OPENWEBUI_LOG}" || true
  false
fi

printf 'testing-web-search\n' >"${STATUS_FILE}"
for _ in $(seq 1 3); do
  if web_search_ok; then
    printf 'ready\n' >"${STATUS_FILE}"
    touch "${STATE_DIR}/ready"
    printf 'vLLM, Open WebUI and keyless DuckDuckGo web search are healthy.\n'
    exit 0
  fi
  sleep 10
done

printf 'DuckDuckGo web search did not return a result after three attempts.\n'
false
