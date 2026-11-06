# End-to-End MLOps Pipeline (Version 3)

A complete end-to-end machine learning pipeline with an interactive Streamlit dashboard,
MLflow experiment tracking, DVC versioning, automated testing, logging, model monitoring,
and Docker containerization.

## Version Comparison

| Feature | Version 1 | Version 2 | Version 3 |
|---------|-----------|-----------|-----------|
| Dataset Loading | ✅ | ✅ | ✅ |
| Data Preprocessing | ✅ | ✅ | ✅ |
| Feature Engineering | ✅ | ✅ | ✅ |
| Model Training | ✅ | ✅ | ✅ |
| Model Evaluation | ✅ | ✅ | ✅ |
| Prediction | ✅ | ✅ | ✅ |
| MLflow Tracking | — | ✅ | ✅ |
| DVC Versioning | — | ✅ | ✅ |
| Logging & Testing | Basic | Advanced | Advanced |
| Streamlit Dashboard | — | — | ✅ |
| Docker Deployment | — | — | ✅ |
| Model Monitoring | — | Basic | ✅ |

## Folder Structure

```
End_to_End_MLOps/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── config/
│   ├── config.py
│   └── params.yaml
│
├── monitoring/
│   ├── __init__.py
│   ├── data_monitor.py
│   └── model_monitor.py
│
├── dashboard/
│   ├── __init__.py
│   ├── overview.py
│   ├── pipeline_ui.py
│   ├── prediction_ui.py
│   └── monitoring_ui.py
│
├── models/
│   └── model.pkl
│
├── outputs/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── reference_data.csv
│   ├── drift_report.html
│   └── performance_report.html
│
├── logs/
│   └── pipeline.log
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_pipeline.py
│
├── mlruns/
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── .gitignore
├── app.py
├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline (Version 2 style)
python main.py

# Run the Streamlit dashboard (Version 3)
streamlit run app.py
```

### Option 2: Run with Docker

```bash
# Build the image
docker build -t mlops-app .

# Run the container
docker run -p 8501:8501 mlops-app

# Or with Docker Compose
docker-compose up --build
```

Then open your browser at: http://localhost:8501

## Dashboard Sections

| Section | Description |
|---------|-------------|
| Overview | Pipeline status, model info, recent logs |
| Dataset | Upload and inspect CSV data |
| Pipeline | Execute the full ML pipeline from UI |
| Evaluation | Metrics, gauge charts, and predictions preview |
| Prediction | Manual entry or batch CSV predictions |
| Monitoring | Data drift detection and model health check |

## MLflow

To view experiment tracking:

```bash
mlflow ui
```

Then open: http://localhost:5000

## Testing

```bash
pytest tests/
```

## DVC

```bash
# Reproduce the pipeline
dvc repro
```

## Monitoring

The dashboard includes:
- **Data drift detection** using Evidently (compares reference vs current data)
- **Model health check** (healthy / warning / critical)
- **Performance degradation** detection (distinguishes drift from accuracy drop)
- **Drift reports** saved as HTML files

**Note:** Data drift does not automatically mean accuracy has decreased.
The system distinguishes data drift (input distribution change) from
actual performance degradation (accuracy drop when labels are available).

## Technologies

- Python, Pandas, NumPy
- Scikit-learn (ML pipeline)
- Joblib (model serialization)
- MLflow (experiment tracking)
- DVC (data/model versioning)
- Streamlit (interactive dashboard)
- Plotly (visualizations)
- Evidently (drift monitoring)
- Pytest (automated testing)
- Docker (containerization)