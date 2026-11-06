"""
Model monitoring module.
Tracks prediction distributions, model accuracy (when labels available),
and distinguishes data drift from actual performance degradation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    PREDICTIONS_PATH,
    METRICS_PATH,
    PERFORMANCE_REPORT_PATH,
    TARGET_COLUMN,
)


def save_prediction_history(results_df: pd.DataFrame, path: str = PREDICTIONS_PATH):
    """Append prediction results to the prediction history file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, results_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        results_df.to_csv(path, index=False)
    print(f"Prediction history saved to {path}")


def compute_performance_summary(predictions_df: pd.DataFrame,
                                metrics_path: str = METRICS_PATH,
                                target_column: str = TARGET_COLUMN):
    """Compute model performance summary from predictions and stored metrics."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_predictions": len(predictions_df),
        "prediction_distribution": {},
        "accuracy_drop": None,
        "performance_degraded": False,
        "drift_without_degradation": False,
    }

    if "prediction" in predictions_df.columns:
        pred_counts = predictions_df["prediction"].value_counts().to_dict()
        summary["prediction_distribution"] = {str(k): int(v) for k, v in pred_counts.items()}

    baseline_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            baseline_metrics = json.load(f)

    has_labels = target_column in predictions_df.columns and "prediction" in predictions_df.columns
    if has_labels:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_true = predictions_df[target_column]
        y_pred = predictions_df["prediction"]
        current_accuracy = accuracy_score(y_true, y_pred)
        current_precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        current_recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        current_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        summary["current_accuracy"] = round(current_accuracy, 4)
        summary["current_precision"] = round(current_precision, 4)
        summary["current_recall"] = round(current_recall, 4)
        summary["current_f1"] = round(current_f1, 4)

        baseline_accuracy = baseline_metrics.get("accuracy")
        if baseline_accuracy is not None:
            accuracy_drop = baseline_accuracy - current_accuracy
            summary["accuracy_drop"] = round(accuracy_drop, 4)
            summary["performance_degraded"] = accuracy_drop > 0.05

    summary["drift_without_degradation"] = (
        summary.get("performance_degraded", False) is False
        and summary.get("accuracy_drop") is not None
    )

    return summary


def save_performance_report(summary: dict, output_path: str = PERFORMANCE_REPORT_PATH):
    """Save model performance summary as JSON report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Performance report saved to {output_path}")


def get_model_health(predictions_df: pd.DataFrame,
                     reference_path: str = PREDICTIONS_PATH,
                     metrics_path: str = METRICS_PATH,
                     target_column: str = TARGET_COLUMN):
    """
    Comprehensive model health check.

    Returns health status: 'healthy', 'warning', or 'critical'.
    - healthy: no drift, no degradation
    - warning: data drift detected but performance stable
    - critical: performance degradation detected
    """
    summary = compute_performance_summary(
        predictions_df, metrics_path=metrics_path, target_column=target_column
    )

    if summary.get("performance_degraded", False):
        status = "critical"
        message = "Model performance degraded. Retraining recommended."
    elif summary.get("drift_without_degradation", False):
        status = "warning"
        message = "Data drift detected but performance stable. Monitor closely."
    else:
        status = "healthy"
        message = "Model healthy. No drift or degradation detected."

    return {
        "status": status,
        "message": message,
        "summary": summary,
    }