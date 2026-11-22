"""
Evaluation metrics for claim verification experiments.
"""

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    results, count_values, classification_type="binary", count_key="total"
):
    """Calculate accuracy, F1 scores and the classification report for each sample count.

    Args:
        results: List of per-claim results (one entry per claim and sample count)
        count_values: Sample counts to evaluate (e.g. [1] or [1, 2, 4, 8, 16, 32])
        classification_type: Only "binary" is supported
        count_key: Key of each result that holds its sample count

    Results whose majority response is not a supported/refuted verdict are skipped.

    Returns:
        Dict of metrics, each one a dict keyed by sample count:
        - accuracy
        - f1_score: macro F1 over the two classes
        - f1_score_binary, precision_binary, recall_binary: supported class (label 0) as positive
        - classification_reports
    """
    metrics = {
        "accuracy": {},
        "f1_score": {},
        "f1_score_binary": {},
        "precision_binary": {},
        "recall_binary": {},
        "classification_reports": {},
    }

    if classification_type != "binary":
        raise ValueError(f"Unsupported classification type: {classification_type}")

    # Text labels for display in reports only
    label_names = ["SUPPORTS", "REFUTES"]

    for count in count_values:
        y_true = []
        y_pred = []

        for result in results:
            if result.get(count_key) == count:
                majority_response = result["majority_response"]

                # Process predicted label: 0=SUPPORTS, 2=REFUTES
                if majority_response in ["0", "SUPPORTS"]:
                    predicted = 0  # SUPPORTS
                elif majority_response in ["2", "REFUTES"]:
                    predicted = 2  # REFUTES
                else:
                    # Skip unknown responses
                    continue

                # Process ground truth
                true_label = result.get("ground_truth")
                if isinstance(true_label, str):
                    if true_label in ["0", "SUPPORTS"]:
                        true_label = 0
                    elif true_label in ["2", "REFUTES"]:
                        true_label = 2
                    else:
                        continue  # Skip if invalid label
                elif isinstance(true_label, int) and true_label in [0, 2]:
                    pass  # Already an int in the right range
                else:
                    continue  # Skip if not a valid numeric label

                y_true.append(true_label)  # Now numeric
                y_pred.append(predicted)  # Now numeric

        if len(y_true) > 0:
            # Calculate accuracy - already using numeric values
            metrics["accuracy"][count] = accuracy_score(y_true, y_pred)

            # Calculate F1-score (macro average)
            metrics["f1_score"][count] = f1_score(y_true, y_pred, average="macro")

            # Calculate binary metrics with SUPPORTS (label 0) as positive class
            metrics["f1_score_binary"][count] = f1_score(y_true, y_pred, pos_label=0)
            metrics["precision_binary"][count] = precision_score(
                y_true, y_pred, pos_label=0
            )
            metrics["recall_binary"][count] = recall_score(y_true, y_pred, pos_label=0)

            # Generate classification report
            metrics["classification_reports"][count] = classification_report(
                y_true,
                y_pred,
                target_names=label_names,  # Use text labels for display only
                output_dict=True,
            )
        else:
            metrics["accuracy"][count] = 0
            metrics["f1_score"][count] = 0
            metrics["f1_score_binary"][count] = 0
            metrics["precision_binary"][count] = 0
            metrics["recall_binary"][count] = 0
            metrics["classification_reports"][count] = {}

    return metrics

