"""
Streamlit dashboard entry point for Version 3.
"""

import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

st.set_page_config(
    page_title="End-to-End MLOps Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.overview import render_overview
from dashboard.pipeline_ui import render_pipeline_ui
from dashboard.prediction_ui import render_prediction_ui
from dashboard.monitoring_ui import render_monitoring_ui


def main():
    st.title("🤖 End-to-End MLOps Dashboard")
    st.markdown("**Version 3** — Deployment, Dashboard & Model Monitoring")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Dataset", "Pipeline", "Evaluation", "Prediction", "Monitoring"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Quick Actions")
    if st.sidebar.button("Run Pipeline"):
        st.session_state["run_pipeline"] = True

    st.sidebar.divider()
    st.sidebar.subheader("About")
    st.sidebar.info(
        "Complete MLOps pipeline with experiment tracking, "
        "versioning, testing, and monitoring."
    )

    if page == "Overview":
        render_overview()
    elif page == "Dataset":
        render_dataset_page()
    elif page == "Pipeline":
        render_pipeline_ui()
    elif page == "Evaluation":
        render_evaluation_page()
    elif page == "Prediction":
        render_prediction_ui()
    elif page == "Monitoring":
        render_monitoring_ui()

    if st.session_state.get("run_pipeline", False):
        st.session_state["run_pipeline"] = False
        st.info("Use the Pipeline page to execute the full pipeline.")


def render_dataset_page():
    st.header("Dataset")
    st.subheader("Upload & Inspect Data")

    from config.config import DATASET_PATH, TARGET_COLUMN
    import pandas as pd

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
            df.to_csv(DATASET_PATH, index=False)
            st.info(f"Dataset saved to {DATASET_PATH}")

            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Data Summary")
            st.dataframe(df.describe(include="all"), use_container_width=True)

            st.subheader("Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.bar_chart(missing)
            else:
                st.info("No missing values found.")

            if TARGET_COLUMN in df.columns:
                st.subheader("Target Distribution")
                st.bar_chart(df[TARGET_COLUMN].value_counts())

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
    else:
        st.info("Upload a CSV file to get started.")
        if os.path.exists(DATASET_PATH):
            st.write(f"Current dataset: {DATASET_PATH}")
            df = pd.read_csv(DATASET_PATH)
            st.dataframe(df.head(10), use_container_width=True)


def render_evaluation_page():
    st.header("Evaluation")
    st.subheader("Model Performance Metrics")

    from config.config import METRICS_PATH, PREDICTIONS_PATH
    import pandas as pd
    import json
    import plotly.graph_objects as go

    if not os.path.exists(METRICS_PATH):
        st.warning("No metrics found. Run the pipeline first.")
        return

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

    st.subheader("Accuracy Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics.get("accuracy", 0),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Model Accuracy"},
        gauge={
            "axis": {"range": [None, 1]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 0.5], "color": "lightgray"},
                {"range": [0.5, 0.75], "color": "yellow"},
                {"range": [0.75, 1.0], "color": "green"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 0.9,
            },
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metrics Comparison")
    fig2 = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        )
    ])
    fig2.update_layout(yaxis=dict(range=[0, 1]), yaxis_title="Score")
    st.plotly_chart(fig2, use_container_width=True)

    if os.path.exists(PREDICTIONS_PATH):
        st.subheader("Recent Predictions")
        pred_df = pd.read_csv(PREDICTIONS_PATH)
        st.dataframe(pred_df.head(20), use_container_width=True)
    else:
        st.info("No predictions file found.")


if __name__ == "__main__":
    main()