#!/usr/bin/env bash
# Table 6: HealthVer, open-book (macro F1).
# Usage: scripts/table6.sh [all|gemini|scicheck_llama|scicheck_qwen]
# The "Reasoning-CV" row is reported by its original authors and has no command.
source "$(dirname "$0")/common.sh"

if row gemini; then          # Gemini
  claim_only healthver_open 30 google "$GEMINI" "$HEALTHVER" healthver
fi
if row scicheck_llama; then  # SciCheck (Llama)
  with_evidence healthver_open 30 "$SCICHECK_PROVIDER" "$SCICHECK_LLAMA" "$HEALTHVER" healthver evidence
fi
if row scicheck_qwen; then   # SciCheck (Qwen)
  with_evidence healthver_open 30 "$SCICHECK_PROVIDER" "$SCICHECK_QWEN" "$HEALTHVER" healthver evidence
fi
