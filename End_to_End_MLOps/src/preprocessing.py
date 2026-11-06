"""
Preprocessing module.
Handles missing values and scales numerical features using Scikit-learn.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def preprocess_data(df: pd.DataFrame, target_column: str = "churn"):
    """
    Handle missing values and scale numerical features.

    Steps:
        1. Separate features and target
        2. Identify numerical and categorical columns
        3. Impute missing values (median for numerical, most_frequent for categorical)
        4. Scale numerical features using StandardScaler

    Args:
        df: Input pandas DataFrame.
        target_column: Name of the target column.

    Returns:
        Tuple of (X_processed, y, imputer, scaler, numerical_features, categorical_features)
    """
    print("[3] Preprocessing completed")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Handle missing values for numerical features
    if numerical_features:
        num_imputer = SimpleImputer(strategy="median")
        X[numerical_features] = num_imputer.fit_transform(X[numerical_features])

    # Handle missing values for categorical features
    if categorical_features:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])

    # Scale numerical features
    if numerical_features:
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
    else:
        scaler = None

    return X, y, num_imputer if numerical_features else None, scaler, numerical_features, categorical_features