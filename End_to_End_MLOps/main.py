"""
Main pipeline entry point for Version 2 (MLOps Automation & Experiment Tracking).
Executes all stages automatically with MLflow tracking, logging, and YAML config.

Stages:
    1. Dataset Loading
    2. Data Validation
    3. Preprocessing
    4. Feature Engineering
    5. Model Training
    6. Model Evaluation
    7. Model Saving
    8. Prediction
"""

import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import load_dataset, validate_data
from preprocessing import preprocess_data
from feature_engineering import engineer_features
from train import train_model
from evaluate import evaluate_model
from predict import load_pipeline, predict_unseen, save_predictions
from config.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
    MIN_SAMPLES_LEAF,
    MODEL_RANDOM_STATE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    LOG_PATH,
)
from logger import pipeline_logger

# MLflow integration
import mlflow
import mlflow.sklearn


def run_pipeline():
    """Execute the complete ML pipeline end-to-end with experiment tracking."""
    pipeline_logger.info("=" * 60)
    pipeline_logger.info("STARTING END-TO-END ML PIPELINE (VERSION 2)")
    pipeline_logger.info("=" * 60)

    # Configure MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="churn_rf_run") as run:
        run_id = run.info.run_id
        pipeline_logger.info(f"MLflow Run ID: {run_id}")

        # Log pipeline parameters to MLflow
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("model_type", "random_forest")

        pipeline_logger.info("Parameters logged to MLflow.")

        # Stage 1: Dataset Loading
        pipeline_logger.info("[1] Loading dataset...")
        df = load_dataset()
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_columns", df.shape[1])

        # Stage 2: Data Validation
        pipeline_logger.info("[2] Validating data...")
        df = validate_data(df)
        mlflow.log_param("missing_values_total", int(df.isnull().sum().sum()))
        mlflow.log_param("duplicate_rows", int(df.duplicated().sum()))

        # Stage 3: Preprocessing
        pipeline_logger.info("[3] Preprocessing data...")
        X, y, imputer, scaler, numerical_features, categorical_features = preprocess_data(
            df, target_column=TARGET_COLUMN
        )
        mlflow.log_param("numerical_features", ",".join(numerical_features))
        mlflow.log_param("categorical_features", ",".join(categorical_features))

        # Stage 4: Feature Engineering
        pipeline_logger.info("[4] Engineering features...")
        X_transformed, encoder, feature_names = engineer_features(
            X, categorical_features, numerical_features
        )
        mlflow.log_param("n_features_final", X_transformed.shape[1])

        # Stage 5: Model Training (includes saving the model)
        pipeline_logger.info("[5] Training model...")
        model, X_test, y_test = train_model(
            X_transformed, y, encoder,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=MODEL_RANDOM_STATE,
        )
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Stage 6: Model Evaluation
        pipeline_logger.info("[6] Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        pipeline_logger.info("Metrics logged to MLflow.")

        # Stage 7: Model Saving (already done in train_model)
        pipeline_logger.info(f"[7] Model saved to models/model.pkl")

        # Stage 8: Prediction on unseen records (using test set as demo)
        pipeline_logger.info("\n" + "=" * 60)
        pipeline_logger.info("GENERATING PREDICTIONS ON UNSEEN RECORDS")
        pipeline_logger.info("=" * 60)

        # Load the saved pipeline
        pipeline = load_pipeline()

        # Use test set as unseen data for demonstration
        unseen_data = X_test.copy()
        unseen_data[TARGET_COLUMN] = y_test.values

        # Generate predictions
        predictions_df = predict_unseen(pipeline, unseen_data, feature_names)

        # Save predictions
        save_predictions(predictions_df)

        pipeline_logger.info("\n" + "=" * 60)
        pipeline_logger.info("PIPELINE EXECUTION COMPLETED")
        pipeline_logger.info("=" * 60)
        pipeline_logger.info("\nSummary:")
        pipeline_logger.info(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        pipeline_logger.info(f"  - Precision: {metrics['precision']:.4f}")
        pipeline_logger.info(f"  - Recall:    {metrics['recall']:.4f}")
        pipeline_logger.info(f"  - F1 Score:  {metrics['f1_score']:.4f}")
        pipeline_logger.info(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
        pipeline_logger.info(f"  - Model saved: models/model.pkl")
        pipeline_logger.info(f"  - Metrics saved: outputs/metrics.json")
        pipeline_logger.info(f"  - Predictions saved: outputs/predictions.csv")
        pipeline_logger.info(f"  - MLflow Run ID: {run_id}")
        pipeline_logger.info(f"  - Logs saved: {LOG_PATH}")


if __name__ == "__main__":
    run_pipeline()