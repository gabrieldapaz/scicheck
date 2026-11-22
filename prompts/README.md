# Prompts

The prompt templates used in the evaluation, stored as plain text. The evaluation code
reads the files exactly as they are (there is no trailing newline), so do not reformat them.

| File | Used by |
|---|---|
| `binary_claim_with_evidence_expert_cot.txt` | SciCheck and "+ Web Search" (claim with evidence) |
| `binary_claim_only_expert_cot.txt` | Gemini, Claim Only and Self-Consistency (claim only) |

Placeholders: `{claim}` and, for the evidence prompt, `{evidence}`.
`--prompt_variant` takes the file name without `.txt`.

The teacher prompt used to generate the reasoning traces is embedded in
`data_preparation/04_label_with_teacher.py`.
