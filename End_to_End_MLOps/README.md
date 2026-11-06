# End-to-End ML Pipeline (Version 1)

A complete end-to-end machine learning pipeline using Python and Scikit-learn.
This version is independently executable — no MLflow, Docker, DVC, or web dashboard required.

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
├── models/
│   └── model.pkl
│
├── outputs/
│   ├── metrics.json
│   └── predictions.csv
│
├── config/
│   └── config.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 01 | Dataset Loading — Load CSV using Pandas |
| 02 | Data Validation — Check missing values, duplicates, data types |
| 03 | Preprocessing — Handle missing values and scale numerical features |
| 04 | Feature Engineering — Encode categorical features |
| 05 | Model Training — Train Random Forest classifier |
| 06 | Model Evaluation — Accuracy, Precision, Recall, F1, ROC-AUC |
| 07 | Model Saving — Save complete pipeline using Joblib |
| 08 | Prediction — Load saved pipeline and predict unseen records |

## Execution

Run the complete pipeline with a single command:

```bash
python main.py
```

## Expected Output

```
============================================================
STARTING END-TO-END ML PIPELINE
============================================================
[1] Dataset loaded successfully
[2] Data validation completed
[3] Preprocessing completed
[4] Feature engineering completed
[5] Model training completed
[6] Model evaluation completed
[7] Model saved successfully
[8] Predictions generated successfully

============================================================
PIPELINE EXECUTION COMPLETED
============================================================
```

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- VS Code