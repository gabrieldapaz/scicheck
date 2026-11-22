"""
Functions for processing claims with language models.
"""

import asyncio
from evaluation.prompts import (
    ClaimOnlyPromptStrategy,
)
from evaluation.utils import logger


async def process_claim(
    i, row, model, strategy, sample_size, counts, prompt_strategy=None
):
    """Process a single claim with the given model and strategy.

    Args:
        i: Index of the claim
        row: DataFrame row containing the claim data
        model: Language model to use
        strategy: Generation strategy to use
        sample_size: Total number of claims being processed
        counts: List of sample counts to test
        prompt_strategy: Strategy for creating prompts (default: ClaimOnlyPromptStrategy)

    Returns:
        List of results for this claim
    """
    claim = row["en_claim"]
    ground_truth_label = row["label"]

    count_type = "Sample Count"

    logger.info(f"Processing claim {i + 1}/{sample_size}: {claim[:50]}...")

    # Use provided prompt strategy or create default one
    if prompt_strategy is None:
        prompt_strategy = ClaimOnlyPromptStrategy()

    # Pre-created prompt, passed to the strategy together with the prompt strategy
    prompt = await prompt_strategy.create_prompt(claim, row)

    claim_results = []

    try:
        results = await strategy.generate(
            model,
            prompt_strategy=prompt_strategy,
            claim=claim,
            row=row,
            prompt=prompt,
        )

        # Add ground truth and claim info
        for count, result in results.items():
            # Use the ground truth label directly (should already be 0, 1, or 2)
            result["ground_truth"] = int(ground_truth_label)
            result["claim"] = claim
            result["claim_id"] = i
            result["input_prompt"] = result.get(
                "input_prompt", "N/A"
            )  # Get prompt from result if available

            # Add correctness evaluation
            is_correct = str(result["majority_response"]) == str(ground_truth_label)
            result["is_correct"] = is_correct

            claim_results.append(result)

            # Log detailed results for each count as soon as they're processed
            logger.info(
                f"------ Claim {i + 1}/{sample_size} Results ({count_type}: {count}) ------"
            )
            logger.info(f"Claim: {claim[:100]}...")
            logger.info(f"Ground Truth: {ground_truth_label}")
            logger.info(
                f"Majority Response: {result['majority_response']} (Votes: {result.get('count', 1)}/{result.get('total', 1)})"
            )

            logger.info(f"Correct: {'yes' if is_correct else 'no'}")

            logger.info(f"Response Distribution: {result['distribution']}")

            if result.get("is_tie", False):
                logger.info(
                    f"Note: Tie detected between {result['tied_responses']}, defaulted to Random"
                )
            logger.info("--------------------------------")
            # Log the first prompt if available
            if len(claim_results) > 0 and "input_prompt" in claim_results[0]:
                logger.info(f"Sample Input Prompt: {claim_results[0]['input_prompt']}")
            logger.info("--------------------------------")

        logger.info(
            f"  Processed claim {i + 1} with {len(counts)} different {count_type.lower()}"
        )
        return claim_results
    except Exception as e:
        logger.error(f"  Error processing claim {i + 1}: {e}")
        return []


async def process_all_claims(
    df, model, strategy, counts, max_concurrent=20, prompt_strategy=None
):
    """Process all claims in parallel with a concurrency limit.

    Args:
        df: DataFrame containing the claims
        model: Language model to use
        strategy: Generation strategy to use
        counts: List of sample counts to test
        max_concurrent: Maximum number of claims to process concurrently
        prompt_strategy: Strategy for creating prompts (default: ClaimOnlyPromptStrategy)

    Returns:
        List of all results
    """
    sample_size = len(df)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(i, row):
        """Process a claim with semaphore control."""
        async with semaphore:
            return await process_claim(
                i, row, model, strategy, sample_size, counts, prompt_strategy
            )

    # Create tasks for all claims with semaphore control
    tasks = [process_with_semaphore(i, row) for i, (_, row) in enumerate(df.iterrows())]

    # Run all tasks and gather results
    results_list = await asyncio.gather(*tasks)

    # Flatten results list
    all_results = []
    for results in results_list:
        all_results.extend(results)

    return all_results
