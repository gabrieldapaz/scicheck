#!/usr/bin/env bash
# Shared settings for the table scripts. Every value can be overridden from the environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# .env is loaded by the Python code; variables exported in the shell take precedence.

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
MLFLOW_FLAG="${MLFLOW_FLAG:---use_mlflow}"          # set to --no-use_mlflow to disable tracking
MAX_CONCURRENT="${MAX_CONCURRENT:-}"                # empty: use the value of each command below
SAMPLE_SIZE="${SAMPLE_SIZE:-}"                      # empty: full dataset
EXTRA_ARGS="${EXTRA_ARGS:-}"                        # appended to every command (e.g. --request_timeout 3600)

# Fine-tuned SciCheck models: the name under which the server exposes each LoRA adapter.
# It must match the adapter name given to vLLM (--lora-modules name=path, see serve_vllm.sh).
SCICHECK_LLAMA="${SCICHECK_LLAMA:-scicheck-llama-3.1-8b}"
SCICHECK_QWEN="${SCICHECK_QWEN:-scicheck-qwen2.5-14b}"
# Provider of the SciCheck models: "local" is the server at SCICHECK_API_BASE
SCICHECK_PROVIDER="${SCICHECK_PROVIDER:-local}"

# Base models (Claim Only, + Web Search, Self-Consistency)
BASE_LLAMA="${BASE_LLAMA:-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo}"
BASE_LLAMA_PROVIDER="${BASE_LLAMA_PROVIDER:-togetherai}"
BASE_QWEN="${BASE_QWEN:-Qwen/Qwen2.5-14B-Instruct:featherless-ai}"
BASE_QWEN_PROVIDER="${BASE_QWEN_PROVIDER:-inference_providers}"

GEMINI="${GEMINI:-gemini-2.5-flash}"

# Evaluation files (versioned in data/)
HEALTHFC="$DATA_DIR/healthfc/healthFC_annotated_tavily_final.csv"
SCIFACT="$DATA_DIR/scifact/scifact_tavily.csv"
COVERT="$DATA_DIR/covert/covert_tavily.csv"
HEALTHVER="$DATA_DIR/healthver/Open_Healthver_test.json"
HEALTHFC_TEST="$DATA_DIR/cer/health_fc_cer_test_tavily.csv"
BIOASQ_TEST="$DATA_DIR/cer/bioasq_cer_test_tavily.csv"

# run_eval <experiment_name> <max_concurrent> <args...>
run_eval() {
  local name="$1" concurrency="$2"; shift 2
  local sample=()
  if [ -n "$SAMPLE_SIZE" ]; then sample=(--sample_size "$SAMPLE_SIZE"); fi
  # shellcheck disable=SC2086
  "$PYTHON" -m evaluation.run_eval \
    --classification_type binary --strategy_type scaling --temperature 0.7 \
    --experiment_name "$name" --max_concurrent "${MAX_CONCURRENT:-$concurrency}" \
    --output_dir "$OUTPUT_DIR" $MLFLOW_FLAG ${sample[@]+"${sample[@]}"} "$@" $EXTRA_ARGS
}

# One generation per claim with the evidence stored in the dataset (SciCheck, + Web Search)
# with_evidence <experiment_name> <max_concurrent> <provider> <model> <dataset> <dataset_type> [evidence_column]
with_evidence() {
  run_eval "$1" "$2" --provider "$3" --model "$4" --dataset "$5" --dataset_type "$6" \
    --sample_counts 0 --prompt_strategy claim_with_static_evidence \
    --prompt_variant binary_claim_with_evidence_expert_cot \
    --static_evidence_column "${7:-evidence_tavily_20_results}"
}

# Claim-only prompt (Gemini, Claim Only, Self-Consistency)
# claim_only <experiment_name> <max_concurrent> <provider> <model> <dataset> <dataset_type> [sample_counts]
claim_only() {
  run_eval "$1" "$2" --provider "$3" --model "$4" --dataset "$5" --dataset_type "$6" \
    --sample_counts "${7:-0}" --prompt_strategy claim_only \
    --prompt_variant binary_claim_only_expert_cot
}

# Run only the rows whose label matches the first script argument (default: all rows)
ROW_FILTER="${1:-all}"
row() { [ "$ROW_FILTER" = all ] || [ "$ROW_FILTER" = "$1" ]; }
