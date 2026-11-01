# Churn MLOps Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17+-0194E2.svg)](https://mlflow.org/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://pytest.org/)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) platform designed to detect customer attrition, evaluate churn risks, and provide interpretable explanations for telecom subscribers. Built on the IBM Telco Customer Churn dataset with Supabase (PostgreSQL) as the persistent cloud data backend.

---

## 📌 Problem Statement

Customer churn is among the most costly problems facing telecommunications and recurring subscription businesses. Acquiring new customers costs 5x to 25x more than retaining existing ones.

This platform solves this by:
1. **Predictive Churn Scoring**: Accurately estimating churn probabilities before cancellations occur.
2. **Handling Severe Class Imbalance**: Solving the ~26.5% minority churn rate via SMOTE and cost-sensitive class weighting inside leakage-free pipelines.
3. **Model Explainability**: Surfacing global and local feature importance via SHAP so business retention teams know *why* a customer is at risk.
4. **Real-time Serving & Monitoring**: Exposing low-latency REST endpoints for inference and a modern Streamlit command dashboard for business stakeholders.

---

## 🏗️ Architecture Overview

```
                                      +-------------------------+
                                      |   data/telco_churn.csv  |
                                      +------------+------------+
                                                   |
                                                   v
                                          [ src/ingest.py ]
                                                   | (Batch Upsert 500)
                                                   v
                                        +---------------------+
                                        |   Supabase (Cloud)  |
                                        |   - customers       |
                                        |   - predictions     |
                                        |   - model_runs      |
                                        +----------+----------+
                                                   |
            +--------------------------------------+--------------------------------------+
            |                                      |                                      |
            v                                      v                                      v
   [ src/train.py ]                        [ api/main.py ]                      [ app/dashboard.py ]
   - Stratified Split                      - GET  /health                       - Executive KPI Cards
   - ImbPipeline + SMOTE                   - POST /predict (Pydantic)           - High-Risk Live Feed
   - Logistic / RF / XGBoost               - GET  /predict/{customer_id}        - Churn Risk Calculator
   - MLflow Tracking & Metrics             - GET  /model-info                   - SHAP Plots
            |                                      |                                      |
            v                                      +--------------------------------------+
   [ src/explain.py ]
   - SHAP Beeswarm & Bar
   - MLflow Artifacts
            |
            v
   models/best_model.joblib
```

### Key Engineering Guardrails
- **Zero Data Leakage**: All imputation (median/mode), scaling (StandardScaler), one-hot encoding, and engineered feature transformations (`TelcoFeatureEngineer`) live strictly inside a scikit-learn / imbalanced-learn `Pipeline`. Data transformations fit *only* on the training partition.
- **Resilient Database Layer**: Typed Supabase data-access module (`src/db.py`) with automatic error handling and logging, designed to run with live Supabase credentials or dry-run offline.
- **Config Via Environment**: No secrets, database keys, or local filepaths are hardcoded. Everything is configured via `.env`.

---

## 📂 Project Structure

```
churn-mlops-platform/
├── data/
│   └── telco_churn.csv           # IBM Telco Customer Churn dataset (gitignored)
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis & statistical insights
├── src/
│   ├── config.py                 # Central configurations, paths, env vars & logging
│   ├── db.py                     # Supabase client & typed database helper methods
│   ├── ingest.py                 # Data cleaning, TotalCharges conversion & Supabase upsert
│   ├── features.py               # Leak-free sklearn Pipeline & feature engineering
│   ├── train.py                  # Model training, SMOTE, Optuna/CV, MLflow tracking
│   ├── explain.py                # SHAP summary & bar feature importance plots
│   └── predict.py                # Inference service & predictions table logger
├── api/
│   └── main.py                   # FastAPI service with health, predict & metadata routes
├── app/
│   └── dashboard.py              # Streamlit interactive operations dashboard
├── supabase/
│   └── schema.sql                # SQL DDL schemas and indexes for Supabase tables
├── tests/
│   ├── conftest.py               # Shared pytest fixtures & mock Supabase client
│   ├── test_features.py          # Unit tests for data cleaning & feature pipelines
│   ├── test_db.py                # Unit tests for Supabase data operations
│   └── test_api.py               # Integration tests for FastAPI endpoints
├── models/                       # Champion models, metadata & plots (gitignored)
├── .env.example                  # Template of required environment variables
├── .env                          # Local environment variables
├── .gitignore                    # Git tracking ignore rules
├── requirements.txt              # Production dependency specifications
└── README.md                     # Documentation & operational handbook
```

---

## 🛠️ Step-by-Step Setup Guide

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool (`venv` or `conda`)
- A free account on [Supabase](https://supabase.com)

### 2. Clone and Create Virtual Environment
```bash
# Navigate to project root
cd "churn-mlops-platform"

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 📋 Manual Steps Required by User

### Step A: Create Supabase Project
1. Go to [database.new](https://database.new) and sign in to Supabase.
2. Click **New project**, choose an organization, give it a name (e.g. `churn-mlops`), and set a database password.
3. Once provisioned, navigate to **Project Settings** > **API**.
4. Copy your **Project URL** and the `service_role` (or `anon` public) **API Key**.

### Step B: Run the Database Schema
1. In the Supabase dashboard, open the **SQL Editor** from the left navigation.
2. Open the file [`supabase/schema.sql`](supabase/schema.sql) in this repository and copy its entire contents.
3. Paste the SQL query into the Supabase SQL Editor and click **Run**.
4. Verify in the **Table Editor** that three tables are created:
   - `customers`
   - `predictions`
   - `model_runs`

### Step C: Configure `.env`
Open `.env` in the project root and fill in your Supabase credentials:
```ini
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-service-role-or-anon-key
```

---

## 🚀 Execution Commands (In Order)

Run each stage in sequence from the project root:

### Stage 1: Ingest Data into Supabase
Cleans raw data, repairs whitespace in `TotalCharges`, maps Churn to boolean, and upserts in batches of 500:
```bash
python src/ingest.py
```
*(Optional dry-run without writing to database: `python src/ingest.py --dry-run`)*

### Stage 2: Train Models & Track with MLflow
Trains Logistic Regression (baseline), Random Forest (with SMOTE), and XGBoost (with SMOTE), performs cross-validation, logs metrics/plots to MLflow, and saves the champion model:
```bash
python src/train.py --tuning-iterations 4
```
*(For rapid smoke testing without hyperparameter search, pass `--no-tune`)*

### Stage 3: Generate SHAP Explainability Plots
Extracts SHAP values and generates global importance beeswarm and bar visualizations:
```bash
python src/explain.py --sample-size 300
```

### Stage 4: Launch MLflow Tracking UI
Inspect experiments, parameter grids, metrics, confusion matrices, and ROC curves:
```bash
mlflow ui --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### Stage 5: Start the FastAPI Service
Launch the low-latency REST inference service:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at [http://localhost:8000/docs](http://localhost:8000/docs).

### Stage 6: Launch the Streamlit Monitoring Dashboard
Launch the interactive web portal for retention teams:
```bash
streamlit run app/dashboard.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Automated Tests

Run the full pytest suite with verbose output:
```bash
pytest tests/ -v
```

All 16 unit and integration tests validate:
- Missing value imputation and TotalCharges whitespace handling
- Zero-leakage sklearn preprocessing pipeline transformations
- Pydantic schema validation and HTTP responses (`/health`, `/predict`, `/model-info`)
- Mocked Supabase customer lookup and inference recording

---

## 🌐 API Reference

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "v1.0.0"
}
```

### 2. Predict Churn for Feature JSON
```http
POST /predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 3.0,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 239.55
}
```
**Response:**
```json
{
  "churn_probability": 0.8842,
  "risk_label": "high",
  "model_version": "v1.0.0",
  "customer_id": null
}
```

### 3. Predict Churn by Customer ID
Fetches stored attributes from Supabase `customers` table and writes the resulting prediction into `predictions`:
```http
GET /predict/7590-VHVEG
```

### 4. Model Info & Metrics
```http
GET /model-info
```

---

## 🚢 Deployment Roadmap (Next Phase)

The codebase has been decoupled from hardcoded paths and secrets to remain 100% deployment-ready.

Planned deployment infrastructure:
1. **Containerization**:
   - `Dockerfile.api`: Minimal multi-stage Python 3.11 Alpine/Slim image running Uvicorn.
   - `Dockerfile.dashboard`: Container running the Streamlit app.
   - `docker-compose.yml`: Multi-container orchestrator connecting API, Dashboard, and local MLflow tracking.
2. **CI/CD Automation (GitHub Actions)**:
   - Linting (`flake8`, `black`, `isort`) & type validation (`mypy`).
   - Automated test runner via `pytest`.
   - Continuous training triggers on data drift detection.
   - Container image build & push to AWS ECR / Google Artifact Registry.
3. **Cloud Hosting**:
   - API: AWS ECS Fargate or Google Cloud Run (auto-scaling serverless containers).
   - Dashboard: Streamlit Community Cloud or Azure App Service.
   - Tracking Server: Hosted MLflow tracking on AWS EC2 or Databricks.
