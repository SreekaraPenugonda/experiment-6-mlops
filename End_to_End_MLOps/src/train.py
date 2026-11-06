"""
Model training module.
Trains a Random Forest classifier and saves the complete pipeline using Joblib.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TEST_SIZE, RANDOM_STATE, MODEL_PATH


def train_model(X: pd.DataFrame, y: pd.Series, encoder=None,
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42):
    """
    Train a Random Forest classifier.

    Args:
        X: Feature DataFrame (processed and engineered).
        y: Target series.
        encoder: Fitted OneHotEncoder (optional, saved for inference consistency).
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (trained_model, X_test, y_test)
    """
    print("[5] Model training completed")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    # Train Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Save the complete pipeline (model + encoder) using Joblib
    pipeline = {
        "model": model,
        "encoder": encoder,
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[7] Model saved successfully to {MODEL_PATH}")

    return model, X_test, y_test