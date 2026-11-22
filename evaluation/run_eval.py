"""
Evaluate a model on a binary claim verification dataset.

This is the single entry point behind every author-run row of Tables 5 to 8:
SciCheck, Gemini, Self-Consistency, Claim Only and + Web Search. See `scripts/`.
"""

import asyncio
import sys
from pathlib import Path

from evaluation.experiment_setup import (
    parse_arguments,
    setup_experiment,
)
from evaluation.claims_processing import (
    process_all_claims,
)
from evaluation.metrics import calculate_metrics
from evaluation.utils import (
    save_results,
    create_experiment_dir,
    save_experiment_metadata,
    logger,
)


async def main():
    """Main function to run the experiment.

    1. Parse command line arguments
    2. Set up the experiment (load dataset, configure model)
    3. Process all claims in parallel
    4. Save results
    5. Calculate metrics
    6. Track the experiment with MLflow (if enabled)
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up MLflow tracking if enabled
    mlflow_tracker = None
    if args.use_mlflow:
        import mlflow
        from evaluation.mlflow_utils import setup_mlflow_tracking

        mlflow.openai.autolog()

        mlflow_tracker = setup_mlflow_tracking(args)
        run_name = args.mlflow_run_name or f"{args.experiment_name}_{args.model}"
        mlflow_tracker.start_run(run_name=run_name)

        # Log experiment parameters
        experiment_params = {
            "model": args.model,
            "provider": args.provider,
            "temperature": args.temperature,
            "sample_counts": args.sample_counts,
            "prompt_strategy": args.prompt_strategy,
            "prompt_variant": args.prompt_variant,
            "sample_size": args.sample_size,
            "max_concurrent": args.max_concurrent,
        }

        # Add static evidence parameters if using static evidence strategy
        if args.prompt_strategy == "claim_with_static_evidence":
            experiment_params.update(
                {
                    "static_evidence_column": args.static_evidence_column,
                }
            )

        mlflow_tracker.log_params(experiment_params)

    # Set up the experiment
    df, model, strategy, sample_counts, prompt_strategy = setup_experiment(args)

    # Process all claims
    all_results = await process_all_claims(
        df,
        model,
        strategy,
        sample_counts,
        max_concurrent=args.max_concurrent,
        prompt_strategy=prompt_strategy,
    )

    # Create a timestamped directory for this experiment run
    output_dir = Path(args.output_dir)
    result_dir = create_experiment_dir(output_dir, args.experiment_name)
    # Save results to the timestamped directory
    results_file = result_dir / f"{args.experiment_name}_results.json"
    save_results(all_results, results_file)

    # Calculate metrics (accuracy and F1-score)
    metrics = calculate_metrics(
        all_results, sample_counts, args.classification_type, count_key="total"
    )

    # Fail instead of reporting a zero metric: an empty classification report
    # means that no claim produced a valid answer for that sample count
    # (for example, every API call failed).
    empty_counts = [
        count for count in sample_counts if not metrics["classification_reports"][count]
    ]
    if empty_counts:
        logger.error(
            f"No valid model response for sample count(s) {empty_counts} "
            f"({len(df)} claims requested). No metrics were written. "
            f"Raw results saved to: {results_file}"
        )
        if args.use_mlflow and mlflow_tracker:
            import mlflow

            mlflow.end_run(status="FAILED")
        sys.exit(1)

    for count in sample_counts:
        evaluated = metrics["classification_reports"][count]["macro avg"]["support"]
        if evaluated < len(df):
            logger.warning(
                f"Sample count {count}: {len(df) - int(evaluated)} of {len(df)} claims "
                f"had no valid answer and are excluded from the metrics"
            )

    print(f"Metrics calculated: {metrics}")

    # Log metrics and artifacts to MLflow if enabled
    if args.use_mlflow and mlflow_tracker:
        # Log metrics
        for count in sample_counts:
            # Get total support from macro avg in classification report
            support = 0
            try:
                if (
                    "classification_reports" in metrics
                    and count in metrics["classification_reports"]
                ):
                    report = metrics["classification_reports"][count]
                    if "macro avg" in report and "support" in report["macro avg"]:
                        support = report["macro avg"]["support"]
            except Exception as e:
                logger.warning(f"Could not get support for count {count}: {e}")

            mlflow_tracker.log_metrics(
                {
                    f"accuracy_{count}": metrics["accuracy"][count],
                    f"f1_score_{count}": metrics["f1_score"][count],
                    f"f1_score_binary_{count}": metrics["f1_score_binary"][count],
                    f"precision_binary_{count}": metrics["precision_binary"][count],
                    f"recall_binary_{count}": metrics["recall_binary"][count],
                    f"support_{count}": support,
                }
            )

        # Log artifacts
        mlflow_tracker.log_artifact(str(results_file))

    # Save experiment metadata
    save_experiment_metadata(
        directory=result_dir,
        args=args,
        metrics={
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "f1_score_binary": metrics["f1_score_binary"],
            "precision_binary": metrics["precision_binary"],
            "recall_binary": metrics["recall_binary"],
        },
        files=[
            results_file.name,
        ],
    )
    # Display results
    logger.info("\nResults:")
    logger.info(f"Accuracy by sample count: {metrics['accuracy']}")
    logger.info(f"F1-score by sample count: {metrics['f1_score']}")
    logger.info(
        f"Binary F1-score (SUPPORTS as positive) by sample count: {metrics['f1_score_binary']}"
    )
    logger.info(
        f"Binary precision (SUPPORTS as positive) by sample count: {metrics['precision_binary']}"
    )
    logger.info(
        f"Binary recall (SUPPORTS as positive) by sample count: {metrics['recall_binary']}"
    )
    logger.info(f"Results saved to: {result_dir}")

    # End MLflow run if active
    if args.use_mlflow and mlflow_tracker:
        mlflow_tracker.end_run()
        logger.info(f"MLflow tracking completed for run: {run_name}")


if __name__ == "__main__":
    asyncio.run(main())
