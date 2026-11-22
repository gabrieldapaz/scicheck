# Data preparation

Builds the training and validation sets used to fine-tune SciCheck. Run the steps in
order. Notebooks expect to be run from this folder, scripts from the repository root.
Input files are described in `../data/README.md`, everything is written to `../data/training/`.
The final output of step 5 is already versioned in `../data/training/`.

| Step | File | What it does | Output |
|---|---|---|---|
| 1 | `01_merge_decontaminate_sample.ipynb` | Loads and merges the source datasets (PubMedQA, BioASQ, COVID-Fact, HealthVer, FEVEROUS, HoVer), removes training instances whose claim or evidence overlaps with the held-out sets in `data/raw/test/`, and draws a label-balanced sample | `sample.jsonl` (3,582 instances) |
| 2 | `02_web_search_tavily.py` | Retrieves web evidence for each claim with Tavily | `sample_tavily.jsonl` |
| 3 | `03_build_variations_and_split.ipynb` | Builds the gold, web search and gold + web search evidence variations and splits claims 90/10 into training and validation | `train_*.jsonl`, `validation_*.jsonl` |
| 4 | `04_label_with_teacher.py` | Generates reasoning and veracity with the teacher (Gemini 2.5 Flash, temperature 0, dynamic thinking) | `*_completions.jsonl` |
| 5 | `05_filter_correct_and_build_sft.ipynb` | Keeps the instances where the teacher verdict matches the gold label and writes the fine-tuning files | `*_correct_only_sft.jsonl` (2,828 training and 319 validation examples) |

```bash
# from the repository root, with TAVILY_API_KEY and GOOGLE_API_KEY set
jupyter nbconvert --to notebook --execute --inplace data_preparation/01_merge_decontaminate_sample.ipynb

python data_preparation/02_web_search_tavily.py \
  --input data/training/sample.jsonl --output data/training/sample_tavily.jsonl \
  --claim_column claim --search_depth basic

jupyter nbconvert --to notebook --execute --inplace data_preparation/03_build_variations_and_split.ipynb

python data_preparation/04_label_with_teacher.py \
  --train_input data/training/train_gold_plus_web_search.jsonl \
  --validation_input data/training/validation_gold_plus_web_search.jsonl \
  --output_dir data/training

jupyter nbconvert --to notebook --execute --inplace data_preparation/05_filter_correct_and_build_sft.ipynb
```

As of November 2026, `gemini-2.5-flash` is not available to new API keys. Only the final
files of step 5 (`*_correct_only_sft.jsonl`, which contain the teacher outputs kept for
training) are versioned in `data/training/`.

Steps 2 and 4 call paid external services whose answers change over time. To rebuild
the exact fine-tuning files of the paper, place the cached outputs of these two steps
(not part of the repository, see `../data/README.md`) in `data/training/` and run
steps 3 and 5.

The same script retrieves the evidence of the evaluation sets, with the advanced
search depth and a publication cut-off date, for example:

```bash
python data_preparation/02_web_search_tavily.py \
  --input data/raw/test/healthFC_annotated.csv \
  --output data/healthfc/healthFC_annotated_tavily_final.csv \
  --claim_column en_claim --search_depth advanced --end_date 2022-12-31
```
