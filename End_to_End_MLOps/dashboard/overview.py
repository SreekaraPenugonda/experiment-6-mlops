"""
Overview dashboard module.
Displays pipeline status and summary metrics.
"""

import streamlit as st
import os
import json
from datetime import datetime

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, sys_path)
from config.config import METRICS_PATH, MODEL_PATH, LOG_PATH


def render_overview():
    """Render the Overview section of the dashboard."""
    st.header("Overview")
    st.subheader("Pipeline Status & Summary")

    col1, col2, col3 = st.columns(3)

    # Model status
    model_exists = os.path.exists(MODEL_PATH)
    with col1:
        st.metric("Model Trained", "Yes" if model_exists else "No")
        if model_exists:
            model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
            st.caption(f"Last trained: {model_mtime.strftime('%Y-%m-%d %H:%M')}")

    # Metrics
    with col2:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        else:
            st.metric("Accuracy", "N/A")
            st.metric("ROC-AUC", "N/A")

    # Log status
    with col3:
        log_exists = os.path.exists(LOG_PATH)
        st.metric("Logging Active", "Yes" if log_exists else "No")
        if log_exists:
            log_mtime = datetime.fromtimestamp(os.path.getmtime(LOG_PATH))
            st.caption(f"Last log entry: {log_mtime.strftime('%Y-%m-%d %H:%M')}")

    st.divider()

    # Recent log entries
    st.subheader("Recent Pipeline Logs")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        recent = lines[-10:] if len(lines) > 10 else lines
        for line in recent:
            st.text(line.strip())
    else:
        st.info("No log file found. Run the pipeline to generate logs.")