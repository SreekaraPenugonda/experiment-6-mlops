"""
Data loading and validation module.
Loads the dataset using Pandas and performs data validation checks:
- Missing values
- Duplicates
- Data types
"""

import pandas as pd
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATASET_PATH, TARGET_COLUMN


def load_dataset(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        filepath: Path to the CSV dataset file.

    Returns:
        Loaded pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[1] Dataset loaded successfully from {filepath}")
        print(f"    Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data validation checks on the dataset.

    Checks:
        - Missing values
        - Duplicate rows
        - Data types of each column

    Args:
        df: Input pandas DataFrame.

    Returns:
        The validated DataFrame.
    """
    print("[2] Data validation completed")

    # Check missing values
    missing_values = df.isnull().sum()
    missing_total = missing_values.sum()
    print(f"    Total missing values: {missing_total}")
    if missing_total > 0:
        print("    Missing values per column:")
        print(missing_values[missing_values > 0].to_string())

    # Check duplicates
    duplicate_count = df.duplicated().sum()
    print(f"    Duplicate rows: {duplicate_count}")

    # Check data types
    print("    Data types:")
    print(df.dtypes.to_string())

    return df