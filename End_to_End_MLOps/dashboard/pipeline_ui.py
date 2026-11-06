"""
Pipeline execution dashboard module.
Provides UI controls to run pipeline stages and view progress.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TARGET_COLUMN, DATASET_PATH


def render_pipeline_ui():
    """Render the Pipeline section of the dashboard."""
    st.header("Pipeline Execution")
    st.subheader("Run End-to-End ML Pipeline")

    # Dataset info
    st.info(f"Dataset path: {DATASET_PATH}")
    if os.path.exists(DATASET_PATH):
        import pandas as pd
        df = pd.read_csv(DATASET_PATH)
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(f"Target column: {TARGET_COLUMN}")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Dataset not found. Please upload data in the Dataset section.")

    st.divider()

    # Pipeline run button
    if st.button("Run Full Pipeline", type="primary", use_container_width=True):
        run_pipeline_from_ui()


def run_pipeline_from_ui():
    """Execute the pipeline and display progress in the UI."""
    import subprocess
    import threading

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    stages = [
        "[1] Dataset Loading",
        "[2] Data Validation",
        "[3] Preprocessing",
        "[4] Feature Engineering",
        "[5] Model Training",
        "[6] Model Evaluation",
        "[7] Model Saving",
        "[8] Prediction",
    ]

    def run_pipeline_thread():
        """Run pipeline in a subprocess and stream output."""
        try:
            # Import and run pipeline directly
            sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, os.path.join(sys_path, "src"))
            from main import run_pipeline

            for i, stage in enumerate(stages):
                status_text.text(f"Running: {stage}...")
                progress_bar.progress((i + 1) / len(stages))

            run_pipeline()
            progress_bar.progress(1.0)
            status_text.text("Pipeline completed successfully!")
            st.success("Pipeline execution completed!")
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Pipeline failed.")

    thread = threading.Thread(target=run_pipeline_thread)
    thread.start()
