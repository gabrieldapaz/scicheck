import asyncio
import re
import random
from typing import Dict, Any, List
from evaluation.utils import logger
from evaluation.models import apply_params_to_model


def map_label_to_number(label):
    """Map text labels to numeric values.

    SUPPORTS -> 0
    NEI -> 1
    REFUTES -> 2
    """
    label = label.upper().strip()
    if label == "SUPPORTS" or label == "SUPPORTED" or label== "TRUE":
        return "0"
    elif label == "NEI":
        return "1"
    elif label == "REFUTES" or label == "REFUTED" or label == "FALSE":
        return "2"


def map_true_false_to_label(response):
    """Map True/False responses to SUPPORTS/REFUTES labels."""
    response = response.upper().strip()
    if response == "TRUE":
        return "SUPPORTS"
    elif response == "FALSE":
        return "REFUTES"
    return response  # Return original if not TRUE or FALSE


def extract_verification(full_response):
    """Extract the verdict from a model response and map it to "0" (supported),
    "1" (NEI) or "2" (refuted). TRUE/FALSE answers are mapped to supported/refuted.
    Returns "UNK" when no verdict is found.
    """
    # Extract the answer between <answer> and </answer> tags after </thought>
    match = re.search(
        r"</thought>.*?<answer>(SUPPORTED|SUPPORTS|NEI|REFUTES|REFUTED|TRUE|FALSE)</answer>",
        full_response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        label = map_true_false_to_label(match.group(1))
        return map_label_to_number(label)

    # Fallback: find any <answer> tag with the labels
    match = re.search(
        r"<answer>(SUPPORTED|SUPPORTS|NEI|REFUTES|REFUTED|TRUE|FALSE)</answer>",
        full_response,
        re.IGNORECASE,
    )
    if match:
        label = map_true_false_to_label(match.group(1))
        return map_label_to_number(label)

    lines = full_response.strip().split("\n")

    # Look for any SUPPORTS, NEI, or REFUTES in the last few lines
    for line in reversed(lines[-20:] if len(lines) > 20 else lines):
        line = line.strip().upper()
        if "SUPPORTS" in line or "TRUE" in line or "SUPPORTED" in line:
            return map_label_to_number("SUPPORTS")
        elif "NEI" in line:
            # Skip NEI to avoid returning "1" - continue looking for other labels
            continue
        elif "REFUTES" in line or "FALSE" in line or "REFUTED" in line:
            return map_label_to_number("REFUTES")

    # No verdict found: the caller retries with a format reminder
    logger.info(
        "Could not extract verification result from response, using retry logic..."
    )
    return "UNK"


class ScalingGenerationStrategy:
    """Generate N completions per claim and aggregate them by majority vote.

    With `generation_counts=[1]` this is a single generation per claim (SciCheck,
    Gemini, Claim Only, + Web Search). With more generations it is the
    Self-Consistency baseline.
    """

    def __init__(self, generation_counts: List[int]):
        """
        Args:
            generation_counts: List of generation counts to evaluate (e.g., [1, 2, 4, 8, 16, 32])
        """
        self.generation_counts = generation_counts

    async def _retry_extraction(
        self, llm, prompt: str, result: str, max_retries: int = 3
    ) -> tuple[str, str]:
        """
        Retry extraction when the initial extraction fails with "UNK".

        Args:
            llm: Language model to use for retries
            prompt: Original prompt
            result: Original result that failed format verification
            max_retries: Maximum number of retry attempts

        Returns:
            tuple: (extracted_result, result) - The extracted verification and the text response
        """
        original_result = result
        retry_llm = apply_params_to_model(llm, {"temperature": 0.0})
        reprompt = """Your response does not follow the specified format. Please reply using exactly one of the permitted options.

        Your response with incorrect format:
        {result}
        
        - If the claim is supported by the evidence, output: <answer>SUPPORTED</answer>
        - If the claim is refuted by the evidence, output: <answer>REFUTED</answer>

        Answer using the correct format:"""
        reprompt = reprompt.format(
            result=original_result
        )

        retries = 0
        extracted_result = "UNK"

        while extracted_result == "UNK" and retries < max_retries:
            logger.info(
                f"  Retry {retries + 1}/{max_retries}: Failed extraction from: {original_result[:100]}..."
            )
            retries += 1
            response = await retry_llm.acomplete(reprompt)
            result = response.text.strip()
            extracted_result = extract_verification(result)
            if extracted_result != "UNK":
                logger.info(f"  Retry successful: extracted label = {extracted_result}")
        if extracted_result == "UNK":
            logger.info(
                "Could not extract verification result from response, using random fallback"
            )
            return random.choice(["0", "2"]), result
        return extracted_result, result

    async def generate(
        self,
        llm,
        prompt_strategy=None,
        claim=None,
        row=None,
        prompt=None,
        max_concurrency: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate responses for each generation count and return results.

        Optimization: Generate only the maximum count needed, then partition results.
        This is much more efficient than generating sum(generation_counts) responses.

        Args:
            llm: Language model to use
            prompt_strategy: Strategy for creating prompts (enables per-generation sampling)
            claim: Claim text (for per-generation prompt creation)
            row: Row data (for per-generation prompt creation)
            prompt: Pre-created prompt (used when no prompt strategy is given)
            max_concurrency: Maximum concurrent generations
        """
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        # Use the pre-created prompt if no prompt strategy is provided
        if prompt_strategy is None and prompt is not None:
            logger.info(f"Using pre-created prompt: {prompt[:100]}...")
        elif prompt_strategy is not None and claim is not None:
            logger.info(
                f"Using prompt strategy for per-generation sampling: {type(prompt_strategy).__name__}"
            )
        else:
            raise ValueError(
                "Either (prompt_strategy + claim) or prompt must be provided"
            )

        async def run_with_semaphore(params=None):
            """Run a single completion with semaphore for concurrency control."""
            async with semaphore:
                try:
                    curr_llm = apply_params_to_model(llm, params)

                    # Create a new prompt for each generation if prompt_strategy is provided
                    if prompt_strategy is not None:
                        current_prompt = await prompt_strategy.create_prompt(claim, row)
                    else:
                        current_prompt = prompt

                    response = await curr_llm.acomplete(current_prompt)
                    result = response.text.strip()
                    logger.info(f"Initial response: {result}...")

                    extracted_result = extract_verification(result)

                    if extracted_result == "UNK":
                        extracted_result, result = await self._retry_extraction(
                            llm, current_prompt, result
                        )

                    return {
                        "response": extracted_result,
                        "raw_response": result,
                        "params": params,
                        "input_prompt": current_prompt,  # Store the prompt used for this generation
                    }
                except Exception as e:
                    print(f"Error during generation: {e}")
                    return {
                        "response": f"ERROR: {str(e)}",
                        "raw_response": f"ERROR: {str(e)}",
                        "params": params,
                        "input_prompt": prompt if prompt_strategy is None else "ERROR",
                    }

        # Generate only the maximum number needed, then partition
        max_generations = max(self.generation_counts)
        print(
            f"  Generating {max_generations} responses (max needed), then partitioning..."
        )

        # Create tasks for maximum count only
        all_tasks = [run_with_semaphore(None) for _ in range(max_generations)]

        # Run all tasks simultaneously and gather all results
        all_results = await asyncio.gather(*all_tasks)

        # Extract responses and parameters
        all_extracted_responses = [result["response"] for result in all_results]
        all_full_responses = [result["raw_response"] for result in all_results]
        all_input_prompts = [result["input_prompt"] for result in all_results]

        # Process all responses at once
        all_processed_responses = all_extracted_responses.copy()

        # Partition responses for each generation count
        results = {}

        for n in self.generation_counts:
            # Take first n responses from the pool (partitioning)
            current_full = all_full_responses[:n]
            current_responses = all_processed_responses[:n]
            current_prompts = all_input_prompts[:n]

            print(
                f"  Processing results for {n} generations (partitioned from {max_generations})..."
            )

            # Count occurrences of each verification result
            response_counts = {}
            for response in current_responses:
                if response in response_counts:
                    response_counts[response] += 1
                else:
                    response_counts[response] = 1

            # Find the most common response
            max_count = 0
            majority_response = None
            tied_responses = []

            for response, count in response_counts.items():
                if count > max_count:
                    max_count = count
                    majority_response = response
                    tied_responses = [response]
                elif count == max_count:
                    tied_responses.append(response)

            is_tie = len(tied_responses) > 1

            if is_tie:
                logger.info(
                    f"Tie detected between {tied_responses}. Falling back to Random."
                )
                majority_response = random.choice(tied_responses)

            results[n] = {
                "majority_response": majority_response,
                "count": max_count,
                "total": n,
                "all_responses": current_responses,
                "raw_responses": current_full,
                "distribution": response_counts,
                "is_tie": is_tie,
                "tied_responses": tied_responses if is_tie else [],
                "params": None,  # always None, kept so the results file keeps its fields
                "input_prompt": (
                    current_prompts[0] if current_prompts else "N/A"
                ),  # Include first prompt as representative
                "all_input_prompts": current_prompts,  # Include all prompts for debugging
            }

        return results
