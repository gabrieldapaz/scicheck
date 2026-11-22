"""
Experiment setup: command line arguments, dataset, model and strategies.
"""

import argparse

from evaluation.models import load_llm
from evaluation.strategies import ScalingGenerationStrategy
from evaluation.utils import load_dataset, logger
from evaluation.config import RESULTS_DIR


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a binary claim verification dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset file (CSV or JSON)",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["healthfc", "scifact", "covert", "healthver", "bioasq"],
        default="healthfc",
        help="Type of dataset being used (affects column names and label preprocessing)",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of claims to sample from the dataset (default: all claims)",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name as exposed by the provider",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=[
            "local",
            "togetherai",
            "google",
            "inference_providers",
        ],
        help="Model provider. 'local' is an OpenAI-compatible server (e.g. vLLM) "
        "at SCICHECK_API_BASE, the others are hosted APIs",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )

    parser.add_argument(
        "--request_timeout",
        type=int,
        default=600,
        help="Request timeout in seconds for model API calls",
    )

    parser.add_argument(
        "--sample_counts",
        type=str,
        default="1,2,4,8",
        help="Either a comma-separated list of sample counts (e.g., '1,2,4,8') or a single integer N to generate powers of 2 up to 2^N (e.g., '8' generates [1,2,4,8,16,32,64,128,256]). Use '0' for a single generation per claim",
    )

    parser.add_argument(
        "--strategy_type",
        type=str,
        choices=["scaling"],
        default="scaling",
        help="Generation strategy (N generations per claim aggregated by majority vote)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save results",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="healthfc_scaling",
        help="Name of the experiment (used for naming output files and directories)",
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum number of claims to process concurrently",
    )

    parser.add_argument(
        "--classification_type",
        type=str,
        choices=["binary"],
        default="binary",
        help="Type of classification: binary (SUPPORTED/REFUTED)",
    )

    parser.add_argument(
        "--use_mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MLflow tracking (default: enabled; pass --no-use_mlflow to disable)",
    )

    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI. If not provided, uses MLFLOW_TRACKING_URI environment variable or local",
    )

    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default=None,
        help="Name for the MLflow run. If not provided, '<experiment_name>_<model>' is used",
    )

    parser.add_argument(
        "--prompt_strategy",
        type=str,
        default="claim_only",
        choices=[
            "claim_only",
            "claim_with_static_evidence",
        ],
        help="Strategy for creating prompts (claim_only or claim_with_static_evidence)",
    )

    parser.add_argument(
        "--prompt_variant",
        type=str,
        required=True,
        help="Name of the prompt template in prompts/ (file name without .txt)",
    )

    parser.add_argument(
        "--static_evidence_column",
        type=str,
        default="evidence_tavily_20_results",
        help="Name of the dataset column containing the evidence (for claim_with_static_evidence)",
    )

    args = parser.parse_args()
    return args


def setup_experiment(args):
    """Set up the experiment based on command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (dataframe, model, strategy, sample_counts, prompt_strategy)
    """
    from evaluation.prompts import (
        ClaimOnlyPromptStrategy,
        ClaimWithStaticEvidenceStrategy,
    )

    # Parse sample counts
    # Check if input is a single integer for powers of 2, or a comma-separated list
    if "," not in args.sample_counts and args.sample_counts.strip().isdigit():
        # Generate powers of 2 from 2^0 up to 2^N
        max_power = int(args.sample_counts.strip())
        sample_counts = [2**i for i in range(max_power + 1)]
    else:
        # Use comma-separated list as before
        sample_counts = [int(x) for x in args.sample_counts.split(",")]

    # Load the dataset with proper classification type
    df = load_dataset(
        args.dataset,
        args.sample_size,
        classification_type=args.classification_type,
        dataset_type=args.dataset_type,
    )

    # Display dataset statistics
    logger.info(f"Dataset loaded: {len(df)} entries")
    logger.info(f"Dataset type: {args.dataset_type}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    logger.info(f"Using classification type: {args.classification_type}")

    # Configure the model
    model = load_llm(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
    )
    logger.info(f"Model loaded: {args.model} (provider: {args.provider})")

    logger.info(
        f"Using ScalingGenerationStrategy with {len(sample_counts)} sample counts"
    )
    strategy = ScalingGenerationStrategy(generation_counts=sample_counts)
    counts_for_evaluation = sample_counts

    # Create the appropriate prompt strategy
    if args.prompt_strategy == "claim_only":
        prompt_strategy = ClaimOnlyPromptStrategy(
            classification_type=args.classification_type,
            prompt_variant=args.prompt_variant,
        )
    elif args.prompt_strategy == "claim_with_static_evidence":
        prompt_strategy = ClaimWithStaticEvidenceStrategy(
            evidence_column=args.static_evidence_column,
            classification_type=args.classification_type,
            prompt_variant=args.prompt_variant,
        )
        logger.info(
            f"Static evidence strategy initialized with evidence_column={args.static_evidence_column}, classification_type={args.classification_type}"
        )
    else:
        raise ValueError(f"Unknown prompt strategy: {args.prompt_strategy}")

    return df, model, strategy, counts_for_evaluation, prompt_strategy
