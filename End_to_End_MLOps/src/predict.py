"""
Prediction module.
Loads the saved pipeline and generates predictions on unseen records.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_PATH, PREDICTIONS_PATH, TARGET_COLUMN


def load_pipeline(model_path: str = MODEL_PATH):
    """
    Load the saved pipeline (model + encoder) from disk.

    Args:
        model_path: Path to the saved model pickle file.

    Returns:
        Dictionary containing the model and encoder.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        sys.exit(1)

    pipeline = joblib.load(model_path)
    print(f"    Model loaded from {model_path}")
    return pipeline


def predict_unseen(pipeline, unseen_data: pd.DataFrame, feature_columns: list):
    """
    Generate predictions on unseen records using the loaded pipeline.

    Args:
        pipeline: Dictionary containing the model and encoder.
        unseen_data: DataFrame with unseen records.
        feature_columns: List of feature column names used during training.

    Returns:
        DataFrame with predictions and probabilities.
    """
    model = pipeline["model"]
    encoder = pipeline["encoder"]

    # Ensure we only use the expected feature columns
    X = unseen_data[feature_columns].copy()

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Build results DataFrame
    results = unseen_data.copy()
    results["prediction"] = predictions
    results["prediction_probability"] = probabilities

    print("[8] Predictions generated successfully")
    return results


def save_predictions(results: pd.DataFrame, output_path: str = PREDICTIONS_PATH):
    """
    Save predictions to a CSV file.

    Args:
        results: DataFrame containing predictions.
        output_path: Path to save the predictions CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"    Predictions saved to {output_path}")