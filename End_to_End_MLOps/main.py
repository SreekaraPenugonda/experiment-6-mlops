"""
Main pipeline entry point.
Executes all stages automatically:
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
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import load_dataset, validate_data
from preprocessing import preprocess_data
from feature_engineering import engineer_features
from train import train_model
from evaluate import evaluate_model
from predict import load_pipeline, predict_unseen, save_predictions
from config.config import TARGET_COLUMN


def run_pipeline():
    """Execute the complete ML pipeline end-to-end."""
    print("=" * 60)
    print("STARTING END-TO-END ML PIPELINE")
    print("=" * 60)

    # Stage 1: Dataset Loading
    df = load_dataset()

    # Stage 2: Data Validation
    df = validate_data(df)

    # Stage 3: Preprocessing
    X, y, imputer, scaler, numerical_features, categorical_features = preprocess_data(
        df, target_column=TARGET_COLUMN
    )

    # Stage 4: Feature Engineering
    X_transformed, encoder, feature_names = engineer_features(
        X, categorical_features, numerical_features
    )

    # Stage 5: Model Training (includes saving the model)
    model, X_test, y_test = train_model(X_transformed, y, encoder)

    # Stage 6: Model Evaluation
    metrics = evaluate_model(model, X_test, y_test)

    # Stage 7: Model Saving (already done in train_model)
    # Stage 8: Prediction on unseen records (using test set as demo)
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS ON UNSEEN RECORDS")
    print("=" * 60)

    # Load the saved pipeline
    pipeline = load_pipeline()

    # Use test set as unseen data for demonstration
    unseen_data = X_test.copy()
    # Add target back for reference
    unseen_data[TARGET_COLUMN] = y_test.values

    # Generate predictions
    predictions_df = predict_unseen(pipeline, unseen_data, feature_names)

    # Save predictions
    save_predictions(predictions_df)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  - Model saved: models/model.pkl")
    print(f"  - Metrics saved: outputs/metrics.json")
    print(f"  - Predictions saved: outputs/predictions.csv")


if __name__ == "__main__":
    run_pipeline()