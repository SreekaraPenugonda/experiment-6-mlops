"""
Model monitoring dashboard module.
Displays data drift and model health information.
"""

import streamlit as st
import pandas as pd
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    DRIFT_REPORT_PATH,
    PERFORMANCE_REPORT_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH,
    REFERENCE_DATA_PATH,
)
from monitoring.data_monitor import check_data_drift
from monitoring.model_monitor import get_model_health, compute_performance_summary


def render_monitoring_ui():
    """Render the Monitoring section of the dashboard."""
    st.header("Monitoring")
    st.subheader("Data Drift & Model Health")

    st.info("""
    **Note:** Data drift does not automatically mean accuracy has decreased.
    We distinguish data drift (input distribution changes) from actual
    performance degradation (accuracy drop when labels are available).
    """)

    # Data drift check
    st.subheader("Data Drift Detection")
    col1, col2 = st.columns([3, 1])
    with col2:
        run_drift = st.button("Run Drift Check", type="primary")

    current_data = None
    if os.path.exists(PREDICTIONS_PATH):
        current_data = pd.read_csv(PREDICTIONS_PATH)

    if run_drift:
        if current_data is not None and os.path.exists(REFERENCE_DATA_PATH):
            with st.spinner("Analyzing data drift..."):
                drift_summary = check_data_drift(
                    current_data,
                    reference_path=REFERENCE_DATA_PATH,
                    output_path=DRIFT_REPORT_PATH,
                )
                st.session_state["drift_summary"] = drift_summary
        else:
            st.warning("Need both reference data and current predictions for drift check.")

    if "drift_summary" in st.session_state:
        ds = st.session_state["drift_summary"]
        if ds.get("error"):
            st.error(ds["error"])
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Drift Detected", "Yes" if ds.get("drift_detected") else "No")
            with col2:
                st.metric("Drifted Columns", f"{ds.get('n_drifted_columns', 0)}/{ds.get('n_total_columns', 0)}")
            with col3:
                st.caption(f"Checked: {ds.get('timestamp', 'N/A')[:19]}")

            if os.path.exists(DRIFT_REPORT_PATH):
                with open(DRIFT_REPORT_PATH, "r") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)

    st.divider()

    # Model health
    st.subheader("Model Health Check")
    if st.button("Check Model Health", type="secondary"):
        if current_data is not None and os.path.exists(METRICS_PATH):
            health = get_model_health(
                current_data,
                metrics_path=METRICS_PATH,
                target_column=TARGET_COLUMN if TARGET_COLUMN in current_data.columns else None,
            )
            st.session_state["model_health"] = health
        else:
            st.warning("Need predictions data and training metrics for health check.")

    if "model_health" in st.session_state:
        health = st.session_state["model_health"]
        status = health["status"]
        message = health["message"]
        summary = health["summary"]

        if status == "critical":
            st.error(f"**{status.upper()}**: {message}")
        elif status == "warning":
            st.warning(f"**{status.upper()}**: {message}")
        else:
            st.success(f"**{status.upper()}**: {message}")

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", summary.get("n_predictions", 0))
        with col2:
            acc = summary.get("current_accuracy")
            st.metric("Current Accuracy", f"{acc:.4f}" if acc else "N/A")
        with col3:
            drop = summary.get("accuracy_drop")
            st.metric("Accuracy Drop", f"{drop:.4f}" if drop is not None else "N/A")
        with col4:
            pred_dist = summary.get("prediction_distribution", {})
            churn_pct = pred_dist.get("1", 0) / max(sum(pred_dist.values()), 1) * 100
            st.metric("Churn Rate", f"{churn_pct:.1f}%")

        # Prediction distribution chart
        if pred_dist:
            st.subheader("Prediction Distribution")
            st.bar_chart(pred_dist)

        # Performance report
        if os.path.exists(PERFORMANCE_REPORT_PATH):
            with open(PERFORMANCE_REPORT_PATH, "r") as f:
                perf = json.load(f)
            st.subheader("Performance Details")
            st.json(perf)
    else:
        st.info("Click 'Check Model Health' to evaluate model status.")