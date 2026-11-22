"""
04. Reasoning and veracity generation with the teacher model (Gemini 2.5 Flash).

Run after notebook 03. Prompts the teacher with each claim and its evidence and
stores the full completion (reasoning trace inside <thought> tags, followed by
the final answer).

    python data_preparation/04_label_with_teacher.py \
        --train_input data/training/train_gold_plus_web_search.jsonl \
        --validation_input data/training/validation_gold_plus_web_search.jsonl \
        --output_dir data/training

Outputs (in --output_dir):
    <input name>_completions.jsonl       all input columns plus `prompt` and `completion`
    <input name>_completions_sft.jsonl   only `prompt` and `completion`

Requires GOOGLE_API_KEY in the environment.
"""

import argparse
import os
import pandas as pd
from openai import AsyncOpenAI
import asyncio
import backoff
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    # Google Gemini API settings (OpenAI-compatible endpoint)
    "gemini_api_key": os.environ.get("GOOGLE_API_KEY"),
    "gemini_model": "gemini-2.5-flash",
    "gemini_api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",

    # Sampling parameters
    "validation_sample_size": 351,
    "train_sample_size": 3231,
    "random_seed": 42,
    
    # API request parameters
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_concurrency": 10,
    
    # Retry parameters
    "max_retries": 5,
    "retry_factor": 2,
    "max_retry_wait": 120,
    "timeout": 360,

    # Prompt template
    "prompt_template": """
<evidence>
{evidence}
</evidence>

You are a professional fact-checker with expertise in verifying claims based on evidence.

Claim: {claim}

First, think step by step and write out your reasoning in a professional, analytical tone.
Then, on a new line, output exactly one of:

- If the claim is supported by the evidence, output: <answer>SUPPORTED</answer>
- If the claim is refuted by the evidence, output: <answer>REFUTED</answer>

No other answer formats will be accepted.

Output:
"""
}

def load_data(file_path):
    return pd.read_json(file_path, lines=True)

def sample_data(df, n, random_seed=None):
    return df.sample(n=n, random_state=random_seed)

def create_prompt(row, template):
    return template.format(
        evidence=row['evidence'],
        claim=row['claim']
    )

@backoff.on_exception(
    backoff.expo,
    (Exception,),
    max_tries=CONFIG["max_retries"],
    factor=CONFIG["retry_factor"],
    jitter=backoff.full_jitter,
    max_value=CONFIG["max_retry_wait"]
)
async def get_completion(prompt):
    """Get completion from the teacher model."""
    client = AsyncOpenAI(
        api_key=CONFIG["gemini_api_key"],
        base_url=CONFIG["gemini_api_base"]
    )
    model = CONFIG["gemini_model"]
    extra_params = {
        "extra_body": {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "include_thoughts": True,
                        "thinking_budget": -1
                    }
                }
            }
        }
    }

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
            timeout=CONFIG["timeout"],
            **extra_params
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        raise

async def get_completions(prompts, max_concurrency=20):
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def get_completion_with_semaphore(prompt_idx, prompt):
        async with semaphore:
            start_time = time.time()
            logger.info(f"Starting request {prompt_idx + 1}")
            try:
                result = await get_completion(prompt)
                elapsed = time.time() - start_time
                logger.info(f"Completed request {prompt_idx + 1} in {elapsed:.1f}s ({len(result or '')} characters)")
                return prompt_idx, result
            except Exception as e:
                logger.error(f"Failed request {prompt_idx + 1}: {str(e)}")
                return prompt_idx, f"ERROR: {str(e)}"
    
    tasks = [get_completion_with_semaphore(i, prompt) for i, prompt in enumerate(prompts)]
    
    # Use asyncio.gather to run all tasks concurrently
    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Sort results back to original order
    results = ["" for _ in range(len(prompts))]
    for task_result in completed_tasks:
        if isinstance(task_result, Exception):
            logger.error(f"Task failed with exception: {task_result}")
        else:
            idx, result = task_result
            results[idx] = result
    
    return results

def output_path(input_path, output_dir, suffix):
    name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{name}{suffix}.jsonl")

async def process_dataset(df, dataset_name, sample_size, random_seed=None):
    sampled_df = sample_data(df, sample_size, random_seed)
    
    logger.info(f"Processing {dataset_name} dataset: {len(sampled_df)} samples")
    logger.info(f"Using teacher model: {CONFIG['gemini_model']}")
    
    prompts = [create_prompt(row, CONFIG["prompt_template"]) for _, row in sampled_df.iterrows()]
    completions = await get_completions(prompts, CONFIG["max_concurrency"])
    
    result_df = sampled_df.copy()
    result_df['prompt'] = prompts
    result_df['completion'] = completions
    
    return result_df

async def run_pipeline(args):

    train_df = load_data(args.train_input)
    validation_df = load_data(args.validation_input)

    logger.info(f"Loaded {len(train_df)} training samples and {len(validation_df)} validation samples")

    train_result = await process_dataset(
        train_df,
        "train",
        CONFIG["train_sample_size"],
        CONFIG["random_seed"]
    )

    validation_result = await process_dataset(
        validation_df,
        "validation",
        CONFIG["validation_sample_size"]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_result.to_json(output_path(args.train_input, args.output_dir, "_completions"), orient='records', lines=True)
    train_result[["prompt", "completion"]].to_json(output_path(args.train_input, args.output_dir, "_completions_sft"), orient='records', lines=True)

    validation_result.to_json(output_path(args.validation_input, args.output_dir, "_completions"), orient='records', lines=True)
    validation_result[["prompt", "completion"]].to_json(output_path(args.validation_input, args.output_dir, "_completions_sft"), orient='records', lines=True)

    logger.info(f"Saved train results: {len(train_result)} samples")
    logger.info(f"Saved validation results: {len(validation_result)} samples")

    return {"train": train_result, "validation": validation_result}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate teacher reasoning and veracity for the training and validation sets")
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--validation_input", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if not CONFIG["gemini_api_key"]:
        raise SystemExit("GOOGLE_API_KEY is not set")
    results = asyncio.run(run_pipeline(args))
    logger.info("Pipeline completed successfully!")
    logger.info(f"Train dataset: {len(results['train'])} completions")
    logger.info(f"Validation dataset: {len(results['validation'])} completions")
