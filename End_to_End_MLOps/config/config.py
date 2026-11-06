"""
Configuration file for the ML pipeline.
Stores dataset path, target column, test size, and random state.
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset configuration
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
TARGET_COLUMN = "churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model output path
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# Output paths
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics.json")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")

# Feature lists (will be populated dynamically)
NUMERICAL_FEATURES = []
CATEGORICAL_FEATURES = []
ALL_FEATURES = []