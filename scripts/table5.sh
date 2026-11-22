#!/usr/bin/env bash
# Table 5: HealthFC, SciFact and CoVERT (macro F1 and binary F1).
# Usage: scripts/table5.sh [all|gemini|sc_llama|sc_qwen|scicheck_llama|scicheck_qwen]
# The "Step-by-Step Verification" row is reported by its original authors and has no command.
source "$(dirname "$0")/common.sh"

if row gemini; then          # Gemini
  claim_only healthfc 10 google "$GEMINI" "$HEALTHFC" healthfc
  claim_only scifact  10 google "$GEMINI" "$SCIFACT"  scifact
  claim_only covert   10 google "$GEMINI" "$COVERT"   covert
fi

# Self-Consistency: 32 generations per claim, majority vote. Report the metrics under key 32.
if row sc_llama; then        # Self-Consistency (Llama)
  claim_only healthfc 10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$HEALTHFC" healthfc 1,2,4,8,16,32
  claim_only scifact  10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$SCIFACT"  scifact  1,2,4,8,16,32
  claim_only covert   10 "$BASE_LLAMA_PROVIDER" "$BASE_LLAMA" "$COVERT"   covert   1,2,4,8,16,32
fi
if row sc_qwen; then         # Self-Consistency (Qwen)
  claim_only healthfc 10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$HEALTHFC" healthfc 1,2,4,8,16,32
  claim_only scifact  10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$SCIFACT"  scifact  1,2,4,8,16,32
  claim_only covert   10 "$BASE_QWEN_PROVIDER" "$BASE_QWEN" "$COVERT"   covert   1,2,4,8,16,32
fi

if row scicheck_llama; then  # SciCheck (Llama)
  with_evidence healthfc 10 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$HEALTHFC" healthfc
  with_evidence scifact  10 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$SCIFACT"  scifact
  with_evidence covert   10 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$COVERT"   covert
fi
if row scicheck_qwen; then   # SciCheck (Qwen)
  with_evidence healthfc 10 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$HEALTHFC" healthfc
  with_evidence scifact  10 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$SCIFACT"  scifact
  with_evidence covert   10 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$COVERT"   covert
fi
