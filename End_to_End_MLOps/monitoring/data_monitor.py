"""
Data drift monitoring module.
Uses Evidently to compare reference vs current data distributions
and detect data drift. Distinguishes data drift from model performance degradation.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    REFERENCE_DATA_PATH,
    DRIFT_REPORT_PATH,
    TARGET_COLUMN,
)


def save_reference_data(df: pd.DataFrame, path: str = REFERENCE_DATA_PATH):
    """
    Save a reference dataset snapshot for drift comparison.

    Args:
        df: Reference DataFrame (typically training data).
        path: Output path for the reference CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Reference data saved to {path}")


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                      output_path: str = DRIFT_REPORT_PATH):
    """
    Detect data drift between reference and current datasets using Evidently.

    Generates an HTML drift report and returns a summary dictionary.

    Args:
        reference_df: Reference (training) DataFrame.
        current_df: Current (incoming/production) DataFrame.
        output_path: Path to save the HTML drift report.

    Returns:
        Dictionary with drift detection results.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        print("[WARNING] Evidently not installed. Install with: pip install evidently")
        return {"drift_detected": False, "error": "evidently not installed"}

    # Drop target column for drift comparison on features only
    ref_features = reference_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    cur_features = current_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    # Align columns
    common_cols = list(set(ref_features.columns) & set(cur_features.columns))
    ref_features = ref_features[common_cols]
    cur_features = cur_features[common_cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)

    # Extract drift summary
    result = report.as_dict()
    n_drifted = 0
    n_features = 0
    drift_status = False
    try:
        for metric in result.get("metrics", []):
            if metric.get("metric") == "DataDriftTable":
                result_data = metric.get("result", {})
                n_drifted = result_data.get("number_of_drifted_columns", 0)
                n_features = result_data.get("number_of_columns", 0)
                drift_status = result_data.get("dataset_drift", False)
                break
    except Exception:
        pass

    summary = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_status,
        "n_drifted_columns": n_drifted,
        "n_total_columns": n_features,
        "report_path": output_path,
    }

    print(f"Data drift report saved to {output_path}")
    print(f"  Drift detected: {drift_status}")
    print(f"  Drifted columns: {n_drifted}/{n_features}")

    return summary


def check_data_drift(current_df: pd.DataFrame,
                     reference_path: str = REFERENCE_DATA_PATH,
                     output_path: str = DRIFT_REPORT_PATH):
    """
    Convenience function: load reference data and check drift against current data.

    Args:
        current_df: Current/production DataFrame.
        reference_path: Path to reference CSV.
        output_path: Path for the HTML drift report.

    Returns:
        Drift summary dictionary.
    """
    if not os.path.exists(reference_path):
        print(f"[WARNING] Reference data not found at {reference_path}")
        return {"drift_detected": False, "error": "no reference data"}

    reference_df = pd.read_csv(reference_path)
    return detect_data_drift(reference_df, current_df, output_path)
