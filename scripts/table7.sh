#!/usr/bin/env bash
# Table 7: HealthFC (Test) and BioASQ-7b (Test) splits used by CER (macro F1).
# Usage: scripts/table7.sh [all|gemini|scicheck_llama|scicheck_qwen]
# The "CER" rows are reported by their original authors and have no command.
source "$(dirname "$0")/common.sh"

if row gemini; then          # Gemini
  claim_only healthfc_test 30 google "$GEMINI" "$HEALTHFC_TEST" healthfc
  claim_only bioasq7b_test 30 google "$GEMINI" "$BIOASQ_TEST"   bioasq
fi
if row scicheck_llama; then  # SciCheck (Llama)
  with_evidence healthfc_test 90 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$HEALTHFC_TEST" healthfc
  with_evidence bioasq7b_test 90 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$BIOASQ_TEST"   bioasq
fi
if row scicheck_qwen; then   # SciCheck (Qwen)
  with_evidence healthfc_test 90 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$HEALTHFC_TEST" healthfc
  with_evidence bioasq7b_test 30 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$BIOASQ_TEST"   bioasq
fi
