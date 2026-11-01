"""Pytest shared test fixtures for churn-mlops-platform."""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_preprocessing_pipeline, get_preprocessing_steps


@pytest.fixture
def sample_customer_dict():
    """Provides a valid single customer feature dictionary."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0,
    }


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic DataFrame mirroring the raw Telco dataset."""
    return pd.DataFrame(
        [
            {
                "customerID": "0001-TESTA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            },
            {
                "customerID": "0002-TESTB",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
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
                "MonthlyCharges": 70.35,
                "TotalCharges": " ",  # whitespace test case
                "Churn": "Yes",
            },
            {
                "customerID": "0003-TESTC",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 45,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 105.65,
                "TotalCharges": "4754.25",
                "Churn": "No",
            },
        ]
    )


@pytest.fixture
def mock_trained_pipeline(sample_raw_dataframe):
    """Provides a fitted dummy ImbPipeline for fast isolated testing."""
    df = sample_raw_dataframe.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    prep_steps = get_preprocessing_steps()
    pipeline = ImbPipeline(
        steps=[
            prep_steps[0],
            prep_steps[1],
            ("classifier", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def mock_supabase_client():
    """Provides a mock Supabase client with chainable table methods."""
    client = MagicMock()
    table_mock = MagicMock()
    client.table.return_value = table_mock

    # Setup chainable methods: upsert, insert, select, eq, limit, order, gte, execute
    query_builder = MagicMock()
    table_mock.upsert.return_value = query_builder
    table_mock.insert.return_value = query_builder
    table_mock.select.return_value = query_builder
    query_builder.eq.return_value = query_builder
    query_builder.limit.return_value = query_builder
    query_builder.order.return_value = query_builder
    query_builder.gte.return_value = query_builder

    # Default execute response
    mock_response = MagicMock()
    mock_response.data = [{"id": 1, "customer_id": "TEST-1234", "features": {"tenure": 10}, "churn_label": False}]
    query_builder.execute.return_value = mock_response

    return client
