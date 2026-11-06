"""
End-to-end pipeline integration tests.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config.config import MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH


def test_model_file_exists():
    """Test that the trained model file exists after pipeline execution."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    print("PASS: test_model_file_exists")


def test_metrics_file_exists():
    """Test that the metrics file exists after pipeline execution."""
    assert os.path.exists(METRICS_PATH), f"Metrics file not found at {METRICS_PATH}"
    print("PASS: test_metrics_file_exists")


def test_metrics_contain_required_keys():
    """Test that metrics JSON contains all required keys."""
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    required_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} is not numeric"
        assert 0 <= metrics[key] <= 1, f"Metric {key} value {metrics[key]} not in [0, 1]"
    print("PASS: test_metrics_contain_required_keys")


def test_predictions_file_exists():
    """Test that the predictions file exists after pipeline execution."""
    assert os.path.exists(PREDICTIONS_PATH), f"Predictions file not found at {PREDICTIONS_PATH}"
    print("PASS: test_predictions_file_exists")


def test_predictions_have_required_columns():
    """Test that predictions CSV contains prediction and probability columns."""
    df = pd.read_csv(PREDICTIONS_PATH)
    assert "prediction" in df.columns, "Predictions missing 'prediction' column"
    assert "prediction_probability" in df.columns, "Predictions missing 'prediction_probability' column"
    assert not df.empty, "Predictions DataFrame is empty"
    print("PASS: test_predictions_have_required_columns")


if __name__ == "__main__":
    test_model_file_exists()
    test_metrics_file_exists()
    test_metrics_contain_required_keys()
    test_predictions_file_exists()
    test_predictions_have_required_columns()
    print("\nAll pipeline tests passed!")
