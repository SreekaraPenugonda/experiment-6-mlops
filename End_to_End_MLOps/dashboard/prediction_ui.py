"""
Prediction dashboard module.
Provides UI for entering data and generating predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    MODEL_PATH, TARGET_COLUMN, PREDICTIONS_PATH,
)
from src.predict import load_pipeline, predict_unseen, save_predictions
from monitoring.model_monitor import save_prediction_history, compute_performance_summary


def render_prediction_ui():
    """Render the Prediction section of the dashboard."""
    st.header("Prediction")
    st.subheader("Generate Predictions on New Data")

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Please run the pipeline first.")
        return

    # Load pipeline to get feature info
    try:
        pipeline = load_pipeline()
        model = pipeline["model"]
        encoder = pipeline["encoder"]
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual entry", "Upload CSV"],
        horizontal=True,
    )

    if input_method == "Manual entry":
        render_manual_prediction(model, encoder)
    else:
        render_csv_prediction(model, encoder)


def render_manual_prediction(model, encoder):
    """Manual single-record prediction form."""
    st.subheader("Single Record Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=24)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)

    with col2:
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col3:
        st.write("")  # spacing
        st.write("")  # spacing
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "gender": gender,
        }])

        try:
            # Need feature names from training - reconstruct from preprocessing
            # For demo, we'll use the columns the model expects
            feature_cols = ["age", "tenure_months", "monthly_charges",
                           "contract_type", "payment_method", "gender"]
            results = predict_unseen(pipeline, input_df, feature_cols)
            prediction = results["prediction"].iloc[0]
            probability = results["prediction_probability"].iloc[0]

            st.divider()
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"Prediction: **CHURN** (Probability: {probability:.4f})")
            else:
                st.success(f"Prediction: **NO CHURN** (Probability: {probability:.4f})")

            # Save prediction
            save_prediction_history(results)
            save_predictions(results)
            st.info("Prediction saved to history.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


def render_csv_prediction(model, encoder):
    """Batch prediction from CSV upload."""
    st.subheader("Batch Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {input_df.shape[0]} rows, {input_df.shape[1]} columns")
            st.dataframe(input_df.head(), use_container_width=True)

            if st.button("Generate Predictions", type="primary"):
                feature_cols = [c for c in input_df.columns if c != TARGET_COLUMN]
                results = predict_unseen(pipeline, input_df, feature_cols)
                st.dataframe(results.head(20), use_container_width=True)

                # Download button
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                save_prediction_history(results)
                save_predictions(results)
                st.info("Predictions saved to history.")

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")