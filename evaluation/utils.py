"""
Dataset loading, label normalization and result persistence.
"""

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("claim_verification")


def load_dataset(
    dataset_path: str,
    sample_size: Optional[int] = None,
    classification_type: str = "binary",
    dataset_type: str = "healthfc",
) -> pd.DataFrame:
    """Load the dataset from a CSV file and filter based on classification type.

    Args:
        dataset_path: Path to the dataset CSV file or JSON file
        sample_size: Number of claims to sample (if None, use all claims)
        classification_type: Type of classification (only 'binary' is used)
        dataset_type: Type of dataset ('healthfc', 'scifact', 'covert', 'healthver', 'bioasq')

    Returns:
        DataFrame containing the claims with normalized labels
    """
    logger.info(f"Loading dataset from {dataset_path} (type: {dataset_type})")
    try:
        # Determine file format and load accordingly
        if dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        # Apply dataset-specific preprocessing
        df = _preprocess_dataset(df, dataset_type)

        # Basic validation after preprocessing
        required_columns = ["en_claim", "label"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Dataset is missing required column: {col}")

        # Convert labels for binary classification if needed
        if classification_type == "binary":
            df = df[df.label.isin(["0", "2", 0, 2])]

        # Sample if requested
        if sample_size is not None:
            if sample_size > len(df):
                logger.warning(
                    f"Requested sample size {sample_size} > dataset size {len(df)}"
                )
                sample_size = len(df)
            df = df.sample(sample_size, random_state=42)
            df = df.reset_index(drop=True)
            logger.info(f"Sampled {sample_size} claims from dataset")

        logger.info(
            f"Loaded {len(df)} claims with label distribution: {df['label'].value_counts().to_dict()}"
        )
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def _preprocess_dataset(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Preprocess dataset based on its type and normalize labels.

    Label normalization:
    - 0: SUPPORTS (TRUE)
    - 1: NEI (NOT ENOUGH INFO)
    - 2: REFUTES (FALSE)

    Args:
        df: Raw dataframe
        dataset_type: Type of dataset to determine preprocessing steps

    Returns:
        Preprocessed dataframe with normalized labels
    """
    logger.info(f"Preprocessing dataset of type: {dataset_type}")

    if dataset_type == "healthfc":
        # HealthFC already uses the correct format, no label mapping needed
        # Ensure we have the right column names
        if "en_claim" not in df.columns:
            raise ValueError("HealthFC dataset missing 'en_claim' column")
        return df

    elif dataset_type == "scifact":
        # SciFact: 1=TRUE (SUPPORTS), 0=FALSE (REFUTES)
        # Map to our format: 1->0 (SUPPORTS), 0->2 (REFUTES)
        logger.info("Applying SciFact label mapping: 1->0 (SUPPORTS), 0->2 (REFUTES)")

        # Ensure we have the right column names
        if "claim" in df.columns and "en_claim" not in df.columns:
            df = df.rename(columns={"claim": "en_claim"})

        # Map labels
        label_mapping = {1: 0, 0: 2}  # SciFact TRUE->SUPPORTS, FALSE->REFUTES
        df["label"] = df["label"].map(label_mapping)

        return df
    elif dataset_type == "covert":
        # COVERT has SUPPORTS AND REFUTES
        # Map to our format: SUPPORTS->0, REFUTES->2, NEI->1
        logger.info("Applying COVERT label mapping: SUPPORTS->0, REFUTES->2")

        # Ensure we have the right column names
        if "claim" in df.columns and "en_claim" not in df.columns:
            df = df.rename(columns={"claim": "en_claim"})

        label_mapping = {"SUPPORTS": 0, "REFUTES": 2}
        df["label"] = df["label"].map(label_mapping)

        return df
    elif dataset_type in ["healthver"]:
        # HealthVer has Supports, Refutes, and Neutral
        # Map to our format: Supports->0, Refutes->2
        # Filter out Neutral (binary classification)
        logger.info("Applying HealthVer label mapping: Supports->0, Refutes->2")

        # Ensure we have the right column names
        if "claim" in df.columns and "en_claim" not in df.columns:
            df = df.rename(columns={"claim": "en_claim"})

        # Filter to only keep Supports and Refutes
        df = df[df["label"].isin(["Supports", "Refutes"])]
        logger.info(
            f"Filtered HealthVer dataset to {len(df)} entries (Supports/Refutes only)"
        )

        # Map labels
        label_mapping = {"Supports": 0, "Refutes": 2}
        df["label"] = df["label"].map(label_mapping)

        return df
    elif dataset_type == "bioasq":
        # BioASQ dataset preprocessing
        # BioASQ uses "yes" and "no" labels
        logger.info("Applying BioASQ label mapping: yes->0 (SUPPORTS), no->2 (REFUTES)")

        # Ensure we have the right column names
        if "claim" in df.columns and "en_claim" not in df.columns:
            df = df.rename(columns={"claim": "en_claim"})

        # Filter to only keep yes and no labels
        df = df[df["label"].isin(["yes", "no"])]
        logger.info(
            f"Filtered BioASQ dataset to {len(df)} entries (yes/no only)"
        )

        # Map labels to our format: yes->0 (SUPPORTS), no->2 (REFUTES)
        label_mapping = {"yes": 0, "no": 2}
        df["label"] = df["label"].map(label_mapping)

        return df
    else:
        logger.warning(f"Unknown dataset type: {dataset_type}. Using as-is.")
        return df


def save_results(results: List[Dict], output_path: Union[str, Path]) -> Path:
    """Save experiment results to a JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path where results will be saved

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_path}")
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return output_path
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """Create a timestamped directory for experiment results.

    Args:
        base_dir: Base directory where to create the experiment directory
        experiment_name: Name of the experiment

    Returns:
        Path to the created directory
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_dir / f"{experiment_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created experiment directory: {result_dir}")
    return result_dir


def save_experiment_metadata(directory: Path, args=None, metrics=None, files=None):
    """Save experiment metadata to a README.md file.

    Args:
        directory: Directory where to save the README
        args: Command line arguments used for the experiment
        metrics: Dictionary of metrics to include
        files: List of files saved in this experiment
    """
    readme_path = directory / "README.md"
    # Directory is "<experiment name>_<YYYYMMDD>_<HHMMSS>"
    experiment_name = directory.name.rsplit("_", 2)[0]

    with open(readme_path, "w") as f:
        f.write(f"# {experiment_name} Experiment\n\n")
        f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Add command line arguments
        if args:
            f.write("## Parameters\n\n")
            f.write("```\n")
            args_dict = vars(args) if hasattr(args, "__dict__") else args
            for key, value in args_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("```\n\n")

        # Add metrics summary if provided
        if metrics:
            f.write("## Results Summary\n\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    f.write(f"### {metric_name}\n")
                    for k, v in metric_value.items():
                        f.write(f"- {k}: {v}\n")
                    f.write("\n")
                else:
                    f.write(f"- {metric_name}: {metric_value}\n")
            f.write("\n")

        # List saved files
        if files:
            f.write("## Files\n\n")
            for file in files:
                f.write(f"- {file}\n")

    logger.info(f"Saved experiment metadata to {readme_path}")
