#!/usr/bin/env bash
# Serve a SciCheck LoRA adapter on top of its base model with vLLM (OpenAI-compatible API).
#
# Usage: scripts/serve_vllm.sh <llama|qwen> <path to adapter directory> [port]
#
# Then point the evaluation to the local server:
#   export SCICHECK_API_BASE=http://127.0.0.1:<port>/v1
#   scripts/table5.sh scicheck_llama
#
# Requires `pip install vllm` and one GPU (about 22 GiB of memory for Llama, 38 GiB for Qwen).
set -euo pipefail
KEY="${1:?usage: serve_vllm.sh <llama|qwen> <adapter dir> [port]}"
ADAPTER="${2:?usage: serve_vllm.sh <llama|qwen> <adapter dir> [port]}"

case "$KEY" in
  llama)
    BASE="${BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
    NAME="${SCICHECK_LLAMA:-scicheck-llama-3.1-8b}"
    PORT="${3:-18081}"; MEMFRAC="${GPU_MEMORY_UTILIZATION:-0.95}" ;;
  qwen)
    BASE="${BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
    NAME="${SCICHECK_QWEN:-scicheck-qwen2.5-14b}"
    PORT="${3:-18082}"; MEMFRAC="${GPU_MEMORY_UTILIZATION:-0.92}" ;;
  *) echo "unknown model key: $KEY (use llama or qwen)" >&2; exit 2 ;;
esac

exec vllm serve "$BASE" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "base-$KEY" \
  --enable-lora --max-lora-rank 64 \
  --lora-modules "$NAME=$ADAPTER" \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization "$MEMFRAC"
