# SciCheck: Reasoning Distillation for Biomedical Claim Verification

Code for the paper **SciCheck: Reasoning Distillation for Biomedical Claim Verification**
(Gabriel Pereira and Luciano Barbosa, SIGIR 2026).

SciCheck verifies a biomedical claim against web evidence with a small language model
(Llama 3.1 8B or Qwen 2.5 14B) fine-tuned with LoRA on reasoning traces distilled from
Gemini 2.5 Flash. This repository contains the data preparation pipeline, the
fine-tuning configuration, the prompts and the evaluation code behind Tables 5 to 8.

```
prompts/            prompt templates used at inference
data_preparation/   training set construction (merge, decontamination, web search, teacher labeling, SFT files)
training/           fine-tuning hyperparameters
evaluation/         evaluation code (single entry point: python -m evaluation.run_eval)
scripts/            one script per table, plus a vLLM launcher for the fine-tuned models
data/               evaluation sets with cached web evidence and the fine-tuning files (see data/README.md)
```

## Installation

Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in the API keys you need
```

All credentials are read from environment variables (or from `.env`):

| Variable | Needed for |
|---|---|
| `SCICHECK_API_BASE` | local server that serves the SciCheck adapters (default `http://127.0.0.1:8000/v1`) |
| `TOGETHERAI_API_KEY` | only for the Llama 3.1 8B base model hosted on Together AI |
| `GOOGLE_API_KEY` | Gemini 2.5 Flash (baseline and teacher). As of November 2026, not available to new API keys. |
| `HF_API_KEY` | Qwen 2.5 14B base model (Hugging Face Inference Providers) |
| `TAVILY_API_KEY` | web evidence retrieval (data preparation only) |

## Data

The evaluation sets, with the web evidence used in the paper, and the final training
and validation files are in this repository under `data/`. Nothing needs to be
downloaded to run the evaluation. `data/README.md` describes every file.

The data comes from third-party datasets. See the original sources for their terms:

- HealthFC: Vladika, Schneider and Matthes, 2024. HealthFC: Verifying Health Claims with Evidence-Based Medical Fact-Checking.
- SciFact: Wadden et al., 2020. Fact or Fiction: Verifying Scientific Claims.
- CoVERT: Mohr, Wührl and Klinger, 2022. CoVERT: A Corpus of Fact-checked Biomedical COVID-19 Tweets.
- HealthVer: Sarrouti et al., 2021. Evidence-based Fact-Checking of Health-related Claims. Open-book evidence from Zheng and Lee, 2025. Reasoning-CV: Fine-tuning Powerful Reasoning LLMs for Knowledge-Assisted Claim Verification.
- HealthFC (Test) and BioASQ-7b (Test) splits: Barone et al., 2025. Combining Evidence and Reasoning for Biomedical Fact-Checking.
- BioASQ: Krithara et al., 2023. BioASQ-QA: A manually curated corpus for Biomedical Question Answering.
- PubMedQA: Jin et al., 2019. PubMedQA: A Dataset for Biomedical Research Question Answering.
- COVID-Fact: Saakyan, Chakrabarty and Muresan, 2021. COVID-Fact: Fact Extraction and Verification of Real-World Claims on COVID-19 Pandemic.
- FEVEROUS: Aly et al., 2021. FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information.
- HoVer: Jiang et al., 2020. HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification.

## Pre-trained models

| Model | Base model | LoRA adapter |
|---|---|---|
| SciCheck (Llama) | `meta-llama/Meta-Llama-3.1-8B-Instruct` | [scicheck-llama-3.1-8b-lora](https://huggingface.co/gabrielpereiramsc/scicheck-llama-3.1-8b-lora) |
| SciCheck (Qwen) | `Qwen/Qwen2.5-14B-Instruct` | [scicheck-qwen2.5-14b-lora](https://huggingface.co/gabrielpereiramsc/scicheck-qwen2.5-14b-lora) |

Serve an adapter locally with vLLM (one GPU) and point the evaluation to it:

```bash
pip install vllm
scripts/serve_vllm.sh llama /path/to/llama-adapter     # or: qwen /path/to/qwen-adapter

export SCICHECK_API_BASE=http://127.0.0.1:18081/v1   # 18082 for qwen
```

## Evaluation

Each script runs the rows of one table that were produced by the authors. Pass a row
name to run a single row, or nothing to run all of them.

```bash
scripts/table5.sh [gemini|sc_llama|sc_qwen|scicheck_llama|scicheck_qwen]
scripts/table6.sh [gemini|scicheck_llama|scicheck_qwen]
scripts/table7.sh [gemini|scicheck_llama|scicheck_qwen]
scripts/table8.sh [llama_claim_only|llama_web_search|llama_sft|qwen_claim_only|qwen_web_search|qwen_sft]
```

A script is a list of calls to the same entry point. For example, SciCheck (Llama) on HealthFC:

```bash
python -m evaluation.run_eval \
  --provider local \
  --model scicheck-llama-3.1-8b \
  --dataset data/healthfc/healthFC_annotated_tavily_final.csv --dataset_type healthfc \
  --prompt_strategy claim_with_static_evidence \
  --prompt_variant binary_claim_with_evidence_expert_cot \
  --static_evidence_column evidence_tavily_20_results \
  --sample_counts 0 --temperature 0.7 --classification_type binary \
  --experiment_name healthfc --max_concurrent 10
```

Each run prints the metrics and writes them, with every model response, to
`results/<experiment_name>_<timestamp>/`. In the output, `f1_score` is the macro F1
over the two classes (column "F1" below) and `f1_score_binary` is the F1 of the
supported class (column "Binary F1"). The paper reports each metric multiplied by 100
and averaged over three runs of the same command. For Self-Consistency, read the
metrics under key `32` (32 generations per claim, majority vote).

Runs are tracked with MLflow by default (`MLFLOW_TRACKING_URI`, or `~/mlruns`).
Use `MLFLOW_FLAG=--no-use_mlflow scripts/table5.sh ...` to disable it. Useful overrides:
`SAMPLE_SIZE=10` for a quick check, `MAX_CONCURRENT`, `OUTPUT_DIR`, and the model names
in `scripts/common.sh`.

## Results

Numbers reported in the paper.

**Table 5.** HealthFC, SciFact and CoVERT.

| Model | HealthFC F1 | HealthFC Binary F1 | SciFact F1 | SciFact Binary F1 | CoVERT F1 | CoVERT Binary F1 | Command |
|---|---|---|---|---|---|---|---|
| Gemini | 77.14 | 83.10 | 79.06 | 84.76 | 69.91 | 80.82 | `scripts/table5.sh gemini` |
| Step-by-Step Verification | - | 81.7 | - | 87.6 | - | 85.9 | reported by the original authors |
| Self-Consistency (Llama) | 71.20 | 80.19 | 71.64 | 79.87 | 62.84 | 69.45 | `scripts/table5.sh sc_llama` |
| Self-Consistency (Qwen) | 73.89 | 79.08 | 67.61 | 72.66 | 73.03 | 81.61 | `scripts/table5.sh sc_qwen` |
| SciCheck (Llama) | 77.72 | 84.03 | 84.12 | 88.76 | 63.88 | 75.54 | `scripts/table5.sh scicheck_llama` |
| SciCheck (Qwen) | 78.17 | 84.35 | 85.43 | 89.07 | 61.43 | 71.64 | `scripts/table5.sh scicheck_qwen` |

**Table 6.** HealthVer (open-book), F1.

| Method | HealthVer | Command |
|---|---|---|
| Gemini | 70.34 | `scripts/table6.sh gemini` |
| Reasoning-CV | 54.05 | reported by the original authors |
| SciCheck (Llama) | 63.77 | `scripts/table6.sh scicheck_llama` |
| SciCheck (Qwen) | 67.00 | `scripts/table6.sh scicheck_qwen` |

**Table 7.** HealthFC and BioASQ-7b test sets, F1.

| Method | HealthFC (Test) | BioASQ-7b (Test) | Command |
|---|---|---|---|
| Gemini | 78.20 | 86.93 | `scripts/table7.sh gemini` |
| CER (in-domain) | 78.03 | 95.20 | reported by the original authors |
| CER (cross-domain) | 75.95 | 89.49 | reported by the original authors |
| SciCheck (Llama) | 76.63 | 84.50 | `scripts/table7.sh scicheck_llama` |
| SciCheck (Qwen) | 78.23 | 85.03 | `scripts/table7.sh scicheck_qwen` |

**Table 8.** Ablation, F1.

| Model | Configuration | HealthFC | SciFact | CoVERT | Command |
|---|---|---|---|---|---|
| Llama 3.1 8B | Claim Only | 68.13 | 69.54 | 61.45 | `scripts/table8.sh llama_claim_only` |
| Llama 3.1 8B | + Web Search | 74.65 | 80.90 | 62.34 | `scripts/table8.sh llama_web_search` |
| Llama 3.1 8B | + Web Search + SFT | 77.72 | 84.12 | 63.88 | `scripts/table8.sh llama_sft` |
| Qwen 2.5 14B | Claim Only | 73.89 | 68.04 | 69.98 | `scripts/table8.sh qwen_claim_only` |
| Qwen 2.5 14B | + Web Search | 76.24 | 87.01 | 70.68 | `scripts/table8.sh qwen_web_search` |
| Qwen 2.5 14B | + Web Search + SFT | 78.17 | 85.43 | 61.43 | `scripts/table8.sh qwen_sft` |

## Training

1. Training set. The files used for fine-tuning are already in the repository:
   `data/training/train_gold_plus_web_search_correct_only_sft.jsonl` (2,828 examples) and
   `data/training/validation_gold_plus_web_search_correct_only_sft.jsonl` (319 examples).
   To rebuild them from the source datasets, run the five numbered steps in
   `data_preparation/` in order (see `data_preparation/README.md`).
2. Fine-tune. Both models are LoRA adapters trained on the Together AI platform with
   the configuration in `training/hyperparameters.yaml` (rank 64, alpha 128, dropout 0.1,
   3 epochs, learning rate 5e-5, batch size 32 for Llama and 8 for Qwen). `training/README.md` has the commands.

## Citation

```bibtex
@inproceedings{10.1145/3805712.3809699,
author = {Pereira, Gabriel and Barbosa, Luciano},
title = {SciCheck: Reasoning Distillation for Biomedical Claim Verification},
year = {2026},
isbn = {9798400725999},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805712.3809699},
doi = {10.1145/3805712.3809699},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1451–1461},
numpages = {11},
keywords = {claim verification, reasoning distillation, retrieval-augmented generation, large language models},
location = {Melbourne, VIC, Australia},
series = {SIGIR '26}
}
```

## Contact

For questions about the code, please open an issue in this repository. For questions about the paper, reach out on [LinkedIn](https://www.linkedin.com/in/marcosgabrielpereira/).

## License

The code is released under the MIT License (see `LICENSE`). Datasets and base models
are third-party work: see the original sources.
