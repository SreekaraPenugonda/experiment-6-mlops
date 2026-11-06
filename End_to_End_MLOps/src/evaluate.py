"""
Model evaluation module.
Calculates performance metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import METRICS_PATH


def evaluate_model(model, X_test, y_test):
    """
    Calculate and save performance metrics.

    Metrics calculated:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC

    Args:
        model: Trained model pipeline.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary of metrics.
    """
    print("[6] Model evaluation completed")

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }

    # Print metrics
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1_score']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Print classification report
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save metrics to JSON
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"    Metrics saved to {METRICS_PATH}")

    return metrics