"""
Feature engineering module.
Transforms input features: encodes categorical variables and prepares features
for model training.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def engineer_features(X: pd.DataFrame, categorical_features: list, numerical_features: list):
    """
    Transform input features for model training.

    Steps:
        1. One-hot encode categorical features
        2. Combine with numerical features
        3. Return final feature matrix

    Args:
        X: Feature DataFrame.
        categorical_features: List of categorical column names.
        numerical_features: List of numerical column names.

    Returns:
        Tuple of (X_transformed, encoder, feature_names)
    """
    print("[4] Feature engineering completed")

    encoder = None
    feature_names = numerical_features.copy()

    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        feature_names.extend(encoded_feature_names.tolist())

        X_num = X[numerical_features].reset_index(drop=True) if numerical_features else pd.DataFrame()
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=X.index)

        if not X_num.empty:
            X_transformed = pd.concat([X_num, X_encoded_df], axis=1)
        else:
            X_transformed = X_encoded_df
    else:
        X_transformed = X[numerical_features].copy()

    return X_transformed, encoder, feature_names