"""
Tests for the preprocessing module.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from preprocessing import preprocess_data


def test_preprocess_returns_expected_outputs():
    """Test that preprocess_data returns the expected number of outputs."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, np.nan],
        "tenure_months": [12, 24, np.nan, 36, 48],
        "monthly_charges": [50.0, 75.0, 100.0, np.nan, 120.0],
        "contract_type": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card", "Electronic check"],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "churn": [0, 1, 0, 1, 0],
    })

    X, y, imputer, scaler, num_features, cat_features = preprocess_data(
        df, target_column="churn"
    )

    assert X is not None, "X should not be None"
    assert y is not None, "y should not be None"
    assert "churn" not in X.columns, "Target column should be removed from X"
    assert len(y) == len(X), "X and y should have the same number of rows"
    assert isinstance(num_features, list), "numerical_features should be a list"
    assert isinstance(cat_features, list), "categorical_features should be a list"
    print("PASS: test_preprocess_returns_expected_outputs")


def test_no_missing_values_after_preprocessing():
    """Test that there are no missing values in X after preprocessing."""
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 50],
        "tenure_months": [12, np.nan, 36, 48, 60],
        "monthly_charges": [50.0, np.nan, 100.0, 120.0, 80.0],
        "contract_type": ["Month-to-month", "One year", "Two year", "Month-to-month", np.nan],
        "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card", "Electronic check"],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "churn": [0, 1, 0, 1, 0],
    })

    X, y, imputer, scaler, num_features, cat_features = preprocess_data(
        df, target_column="churn"
    )

    missing_count = X.isnull().sum().sum()
    assert missing_count == 0, f"X still contains {missing_count} missing values after preprocessing"
    print("PASS: test_no_missing_values_after_preprocessing")


def test_numerical_features_scaled():
    """Test that numerical features are scaled (approximately zero mean, unit variance)."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 50],
        "tenure_months": [12, 24, 36, 48, 60],
        "monthly_charges": [50.0, 75.0, 100.0, 120.0, 80.0],
        "contract_type": ["One year", "One year", "Two year", "Two year", "One year"],
        "payment_method": ["Bank transfer", "Bank transfer", "Credit card", "Credit card", "Bank transfer"],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "churn": [0, 1, 0, 1, 0],
    })

    X, y, imputer, scaler, num_features, cat_features = preprocess_data(
        df, target_column="churn"
    )

    if num_features:
        for col in num_features:
            col_mean = X[col].mean()
            col_std = X[col].std(ddof=0)  # sklearn StandardScaler uses ddof=0 (population std)
            assert abs(col_mean) < 0.1, f"Column {col} mean ({col_mean}) not close to 0 after scaling"
            assert abs(col_std - 1.0) < 0.1, f"Column {col} std ({col_std}) not close to 1 after scaling"
    print("PASS: test_numerical_features_scaled")


if __name__ == "__main__":
    test_preprocess_returns_expected_outputs()
    test_no_missing_values_after_preprocessing()
    test_numerical_features_scaled()
    print("\nAll preprocessing tests passed!")
