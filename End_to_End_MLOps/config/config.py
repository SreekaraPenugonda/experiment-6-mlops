"""
Configuration file for the ML pipeline.
Loads settings from params.yaml and exposes them as module-level constants.
Falls back to sensible defaults if the YAML file is missing.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to load YAML configuration
PARAMS_PATH = os.path.join(BASE_DIR, "config", "params.yaml")

try:
    import yaml

    with open(PARAMS_PATH, "r") as _f:
        _params = yaml.safe_load(_f)

    # Data configuration
    DATASET_PATH = os.path.join(BASE_DIR, _params["data"]["dataset_path"])
    TARGET_COLUMN = _params["data"]["target_column"]
    TEST_SIZE = _params["data"]["test_size"]
    RANDOM_STATE = _params["data"]["random_state"]

    # Preprocessing configuration
    NUMERICAL_IMPUTE_STRATEGY = _params["preprocessing"]["numerical_impute_strategy"]
    CATEGORICAL_IMPUTE_STRATEGY = _params["preprocessing"]["categorical_impute_strategy"]
    SCALER_TYPE = _params["preprocessing"]["scaler"]

    # Feature engineering configuration
    ENCODE_CATEGORICAL = _params["feature_engineering"]["encode_categorical"]
    HANDLE_UNKNOWN = _params["feature_engineering"]["handle_unknown"]

    # Model configuration
    MODEL_TYPE = _params["model"]["type"]
    N_ESTIMATORS = _params["model"]["n_estimators"]
    MAX_DEPTH = _params["model"]["max_depth"]
    MIN_SAMPLES_SPLIT = _params["model"]["min_samples_split"]
    MIN_SAMPLES_LEAF = _params["model"]["min_samples_leaf"]
    MODEL_RANDOM_STATE = _params["model"]["random_state"]

    # MLflow configuration
    MLFLOW_EXPERIMENT_NAME = _params["mlflow"]["experiment_name"]
    MLFLOW_TRACKING_URI = _params["mlflow"]["tracking_uri"]
    MLFLOW_REGISTERED_MODEL_NAME = _params["mlflow"]["registered_model_name"]

    # Output paths
    MODEL_PATH = os.path.join(BASE_DIR, _params["outputs"]["model_path"])
    METRICS_PATH = os.path.join(BASE_DIR, _params["outputs"]["metrics_path"])
    PREDICTIONS_PATH = os.path.join(BASE_DIR, _params["outputs"]["predictions_path"])
    LOG_PATH = os.path.join(BASE_DIR, _params["outputs"]["log_path"])

    # Monitoring paths
    MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")
    REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "outputs", "reference_data.csv")
    DRIFT_REPORT_PATH = os.path.join(BASE_DIR, "outputs", "drift_report.html")
    PERFORMANCE_REPORT_PATH = os.path.join(BASE_DIR, "outputs", "performance_report.html")

except Exception:
    # Fallback defaults if params.yaml is unavailable
    DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
    TARGET_COLUMN = "churn"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    NUMERICAL_IMPUTE_STRATEGY = "median"
    CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
    SCALER_TYPE = "standard"

    ENCODE_CATEGORICAL = True
    HANDLE_UNKNOWN = "ignore"

    MODEL_TYPE = "random_forest"
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    MODEL_RANDOM_STATE = 42

    MLFLOW_EXPERIMENT_NAME = "churn_prediction_pipeline"
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    MLFLOW_REGISTERED_MODEL_NAME = "churn_rf_model"

    MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics.json")
    PREDICTIONS_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")
    LOG_PATH = os.path.join(BASE_DIR, "logs", "pipeline.log")

    # Monitoring paths
    MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")
    REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "outputs", "reference_data.csv")
    DRIFT_REPORT_PATH = os.path.join(BASE_DIR, "outputs", "drift_report.html")
    PERFORMANCE_REPORT_PATH = os.path.join(BASE_DIR, "outputs", "performance_report.html")

# Feature lists (populated dynamically at runtime)
NUMERICAL_FEATURES = []
CATEGORICAL_FEATURES = []
ALL_FEATURES = []