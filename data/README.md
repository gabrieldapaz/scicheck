# Data

## Evaluation sets (versioned, used by `scripts/table*.sh`)

| Path | Rows used | Table | Loaded with |
|---|---|---|---|
| `healthfc/healthFC_annotated_tavily_final.csv` | 327 | 5, 8 | `--dataset_type healthfc` |
| `scifact/scifact_tavily.csv` | 693 | 5, 8 | `--dataset_type scifact` |
| `covert/covert_tavily.csv` | 264 | 5, 8 | `--dataset_type covert` |
| `healthver/Open_Healthver_test.json` | 1,096 | 6 | `--dataset_type healthver`, evidence column `evidence` |
| `cer/health_fc_cer_test_tavily.csv` | 65 | 7 | `--dataset_type healthfc` |
| `cer/bioasq_cer_test_tavily.csv` | 149 | 7 | `--dataset_type bioasq` |

Each CSV is the original dataset plus the cached Tavily search results as formatted
text: `evidence_tavily_20_results` (used at evaluation) and `evidence_tavily_5_results`.
The raw search responses are not included. The evidence can be retrieved again from the
original datasets with `data_preparation/02_web_search_tavily.py`.
`Open_Healthver_test.json` is the HealthVer test split with the open-book evidence
released with Reasoning-CV.

## Fine-tuning files (versioned)

| Path | Examples |
|---|---|
| `training/train_gold_plus_web_search_correct_only_sft.jsonl` | 2,828 |
| `training/validation_gold_plus_web_search_correct_only_sft.jsonl` | 319 |

One JSON object per line with `prompt` and `completion`.

## Inputs of the data preparation (not versioned)

Only needed to rebuild the fine-tuning files with `data_preparation/`.
These files are not part of the repository because of their size. Open an issue if you need them. Place the files as below.

`raw/train/`

| File | Source |
|---|---|
| `ori_pqal.json` | PubMedQA, expert-labeled subset |
| `train-00000-of-00001.parquet` | PubMedQA, artificial subset |
| `training12b_new.json` | BioASQ Task 12b training set |
| `COVIDFACT_dataset.jsonl` | COVID-Fact |
| `Healthver_train.json`, `Feverous_train.json`, `Hover_train.json` | HealthVer, FEVEROUS and HoVer training files released with Reasoning-CV |

`raw/test/` (held-out sets, used only to remove overlapping training instances)

| File | Source |
|---|---|
| `healthFC_annotated.csv` | HealthFC without web evidence (the version with web evidence is read from `healthfc/healthFC_annotated_tavily_final.csv`) |
| `scifact_no-nei_dataset.csv`, `Scifact_dev.json`, `Scifact_train.json` | SciFact |
| `CoVERT_FC_annotations.jsonl` | CoVERT |
| `12B1_golden.json` to `12B4_golden.json` | BioASQ Task 12b test batches |
| `Healthver_test.json`, `Open_Healthver_test.json` | HealthVer test split, gold and open-book evidence |
| `FEVEROUS.json`, `Open_FEVEROUS.json`, `HOVER.json`, `Open_HOVER.json` | FEVEROUS and HoVer evaluation files |
| `VitaminC_dev.json`, `VitaminC_test.json` | VitaminC |
| `LLM-AggreFact_test.json` | LLM-AggreFact |

`training/` also receives the intermediate files of the data preparation
(`sample.jsonl`, `sample_tavily.jsonl`, `*_completions.jsonl`, and so on). The cached
outputs of the two paid steps (web search and teacher labeling) are not part of the
repository because of their size. Open an issue if you need them.
