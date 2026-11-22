#!/usr/bin/env bash
# Table 8: ablation on HealthFC, SciFact and CoVERT (macro F1).
# Usage: scripts/table8.sh [all|llama_claim_only|llama_web_search|llama_sft|qwen_claim_only|qwen_web_search|qwen_sft]
source "$(dirname "$0")/common.sh"

if row llama_claim_only; then  # Llama 3.1 8B, Claim Only
  claim_only healthfc 10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$HEALTHFC" healthfc
  claim_only scifact  10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$SCIFACT"  scifact
  claim_only covert   10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$COVERT"   covert
fi
if row llama_web_search; then  # Llama 3.1 8B, + Web Search
  with_evidence healthfc 10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$HEALTHFC" healthfc
  with_evidence scifact  10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$SCIFACT"  scifact
  with_evidence covert   10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$COVERT"   covert
fi
if row llama_sft; then         # Llama 3.1 8B, + Web Search + SFT (SciCheck)
  with_evidence healthfc 10 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$HEALTHFC" healthfc
  with_evidence scifact  10 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$SCIFACT"  scifact
  with_evidence covert   10 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$COVERT"   covert
fi

if row qwen_claim_only; then   # Qwen 2.5 14B, Claim Only
  claim_only healthfc 10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$HEALTHFC" healthfc
  claim_only scifact  10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$SCIFACT"  scifact
  claim_only covert   10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$COVERT"   covert
fi
if row qwen_web_search; then   # Qwen 2.5 14B, + Web Search
  with_evidence healthfc 10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$HEALTHFC" healthfc
  with_evidence scifact  10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$SCIFACT"  scifact
  with_evidence covert   10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$COVERT"   covert
fi
if row qwen_sft; then          # Qwen 2.5 14B, + Web Search + SFT (SciCheck)
  with_evidence healthfc 10 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$HEALTHFC" healthfc
  with_evidence scifact  10 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$SCIFACT"  scifact
  with_evidence covert   10 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$COVERT"   covert
fi
