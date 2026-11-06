"""
Tests for the data loading and validation module.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_loader import load_dataset, validate_data


def test_load_dataset():
    """Test that the dataset loads successfully and is non-empty."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame), "Loaded object is not a DataFrame"
    assert not df.empty, "DataFrame is empty"
    assert df.shape[0] > 0, "DataFrame has no rows"
    assert df.shape[1] > 0, "DataFrame has no columns"
    print("PASS: test_load_dataset")


def test_validate_data_returns_dataframe():
    """Test that validate_data returns the same DataFrame."""
    df = load_dataset()
    result = validate_data(df)
    assert isinstance(result, pd.DataFrame), "validate_data did not return a DataFrame"
    assert result.shape == df.shape, "validate_data changed the DataFrame shape"
    print("PASS: test_validate_data_returns_dataframe")


def test_no_duplicate_rows():
    """Test that the dataset has no duplicate rows after validation."""
    df = load_dataset()
    duplicate_count = df.duplicated().sum()
    assert duplicate_count == 0, f"Dataset contains {duplicate_count} duplicate rows"
    print("PASS: test_no_duplicate_rows")


def test_target_column_exists():
    """Test that the target column exists in the dataset."""
    df = load_dataset()
    assert "churn" in df.columns, "Target column 'churn' not found in dataset"
    print("PASS: test_target_column_exists")


if __name__ == "__main__":
    test_load_dataset()
    test_validate_data_returns_dataframe()
    test_no_duplicate_rows()
    test_target_column_exists()
    print("\nAll data tests passed!")
