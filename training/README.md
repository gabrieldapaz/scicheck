# Training

Both SciCheck models are LoRA adapters trained on the [Together AI](https://www.together.ai/)
fine-tuning platform from the prompt/completion files produced by `data_preparation/`.
There is no local training script: the job is configured on the platform with the
values in [`hyperparameters.yaml`](hyperparameters.yaml).

| | Llama 3.1 8B Instruct | Qwen 2.5 14B Instruct |
|---|---|---|
| LoRA rank / alpha / dropout | 64 / 128 / 0.1 | 64 / 128 / 0.1 |
| Trainable modules | all linear (q, k, v, o, up, gate, down projections) | same |
| Epochs | 3 | 3 |
| Learning rate, scheduler | 5e-5, cosine (0.5 cycles), warmup ratio 0 | same |
| Batch size | 32 | 8 |
| Max gradient norm, weight decay | 1, 0 | same |
| Loss | cross-entropy on the completion only (prompt masked) | same |

## Launching the jobs

With the Together CLI (`pip install together`), upload the two files and create one
job per base model. The CLI reads its key from `TOGETHER_API_KEY` (the evaluation code
of this repository reads `TOGETHERAI_API_KEY`). The command below is illustrative of
the job parameters:

```bash
together files upload data/training/train_gold_plus_web_search_correct_only_sft.jsonl
together files upload data/training/validation_gold_plus_web_search_correct_only_sft.jsonl

together fine-tuning create \
  --training-file <train file id> --validation-file <validation file id> \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
  --lora --lora-r 64 --lora-alpha 128 --lora-dropout 0.1 \
  --n-epochs 3 --n-evals 3 --batch-size 32 --learning-rate 5e-5 \
  --lr-scheduler-type cosine --warmup-ratio 0 --max-grad-norm 1 --weight-decay 0 \
  --suffix scicheck
```

Repeat with `--model Qwen/Qwen2.5-14B-Instruct --batch-size 8` for the second model.

When a job finishes, download the adapter (`together fine-tuning download <job id>`)
to serve it locally with `scripts/serve_vllm.sh`, or call it through the Together API.
