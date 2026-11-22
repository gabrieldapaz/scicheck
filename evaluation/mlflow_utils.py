"""
MLflow utilities for experiment tracking in claim verification experiments.
"""

import os
import mlflow
from typing import Dict, Any, Optional
import numpy as np
from evaluation.utils import logger


class MLflowExperimentTracker:
    """MLflow-based experiment tracker for claim verification experiments."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        create_if_not_exists: bool = True,
    ):
        """Initialize MLflow experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (if None, uses MLFLOW_TRACKING_URI env var or local)
            create_if_not_exists: Whether to create the experiment if it doesn't exist
        """
        # Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self.run_id = None

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None and create_if_not_exists:
            logger.info(f"Creating new MLflow experiment: {experiment_name}")
            mlflow.create_experiment(experiment_name)
        elif experiment is None and not create_if_not_exists:
            raise ValueError(
                f"Experiment '{experiment_name}' does not exist and create_if_not_exists is False"
            )

        self.experiment_id = mlflow.get_experiment_by_name(
            experiment_name
        ).experiment_id

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start an MLflow run.

        Args:
            run_name: Optional name for the run

        Returns:
            run_id: The ID of the started run
        """
        run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        self.run_id = run.info.run_id
        logger.info(f"Started MLflow run: {self.run_id} (name: {run_name})")
        return self.run_id

    def end_run(self):
        """End the current MLflow run."""
        if self.run_id:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")
            self.run_id = None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")

        # Handle nested dictionaries by flattening them
        flattened_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)):
                        flattened_params[f"{key}.{nested_key}"] = nested_value
            elif isinstance(value, (str, int, float, bool)):
                flattened_params[key] = value

        mlflow.log_params(flattened_params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step for the metrics
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")

        # Handle nested dictionaries or dictionaries with array values
        flattened_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        flattened_metrics[f"{key}.{nested_key}"] = nested_value
                    elif isinstance(nested_value, (list, np.ndarray)):
                        for i, v in enumerate(nested_value):
                            if isinstance(v, (int, float)):
                                flattened_metrics[f"{key}.{nested_key}.{i}"] = v
            elif isinstance(value, (int, float)):
                flattened_metrics[key] = value
            elif isinstance(value, (list, np.ndarray)):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        flattened_metrics[f"{key}.{i}"] = v

        mlflow.log_metrics(flattened_metrics, step=step)

    def log_artifact(self, local_path: str):
        """Log an artifact to MLflow.

        Args:
            local_path: Path to the artifact file
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")

        mlflow.log_artifact(local_path)


def setup_mlflow_tracking(args):
    """Set up MLflow tracking based on command line arguments.

    Args:
        args: Command line arguments with experiment_name and optional mlflow_tracking_uri

    Returns:
        MLflowExperimentTracker: Configured experiment tracker
    """
    tracking_uri = getattr(args, "mlflow_tracking_uri", None)
    experiment_name = getattr(args, "experiment_name", "claim_verification_experiment")

    # Check for environment variable if not specified in args
    if not tracking_uri:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # Set a default tracking URI if none provided (an empty value counts as not provided)
    if not tracking_uri:
        # Use mlruns directory in user's home
        home_dir = os.path.expanduser("~")
        tracking_uri = f"file://{home_dir}/mlruns"

    # Make sure the URI has the proper format
    if not any(
        tracking_uri.startswith(prefix) for prefix in ["http://", "https://", "file://"]
    ):
        tracking_uri = f"file://{os.path.abspath(tracking_uri)}"

    logger.info(f"Using MLflow tracking URI: {tracking_uri}")

    return MLflowExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
