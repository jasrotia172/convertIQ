"""Probability metrics and validation-only decision-policy selection."""

import numpy as np
from sklearn import metrics


def evaluate(y, probabilities, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(probabilities)
    pred = p >= threshold
    result = {
        "accuracy": metrics.accuracy_score(y, pred),
        "precision": metrics.precision_score(y, pred, zero_division=0),
        "recall": metrics.recall_score(y, pred, zero_division=0),
        "f1": metrics.f1_score(y, pred, zero_division=0),
        "balanced_accuracy": metrics.balanced_accuracy_score(y, pred),
        "log_loss": metrics.log_loss(y, p, labels=[0, 1]),
        "brier_score": metrics.brier_score_loss(y, p),
        "roc_auc": metrics.roc_auc_score(y, p) if len(np.unique(y)) == 2 else None,
        "average_precision": metrics.average_precision_score(y, p) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": metrics.confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    return result


def threshold_table(y, probabilities, fp_cost: float = 1, fn_cost: float = 5) -> list[dict]:
    rows = []
    for threshold in np.linspace(0, 1, 101):
        predicted = np.asarray(probabilities) >= threshold
        tn, fp, fn, tp = metrics.confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "illustrative_cost": float(fp * fp_cost + fn * fn_cost),
            }
        )
    return rows


def choose_threshold(y, probabilities, minimum_precision: float = 0.3) -> float:
    rows = threshold_table(y, probabilities)
    eligible = [row for row in rows if row["precision"] >= minimum_precision]
    if not eligible:
        raise ValueError("No validation threshold satisfies the configured minimum precision.")
    return max(eligible, key=lambda row: (row["f1"], row["precision"]))["threshold"]
