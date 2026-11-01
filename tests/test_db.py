"""Unit tests for Supabase data access layer with mocked client."""

from unittest.mock import MagicMock
import pytest
from src.db import (
    get_customer_features,
    get_recent_predictions,
    insert_model_run,
    insert_prediction,
    upsert_customers,
)


def test_upsert_customers_mock(mock_supabase_client):
    """Verifies customer batch upsert executes table upsert on conflict customer_id."""
    records = [
        {"customer_id": "CUST-001", "features": {"tenure": 5}, "churn_label": False},
        {"customer_id": "CUST-002", "features": {"tenure": 12}, "churn_label": True},
    ]

    mock_resp = MagicMock()
    mock_resp.data = records
    mock_supabase_client.table().upsert().execute.return_value = mock_resp

    count = upsert_customers(records, client=mock_supabase_client)
    assert count == 2
    mock_supabase_client.table.assert_called_with("customers")


def test_get_customer_features_mock(mock_supabase_client):
    """Verifies customer features retrieval by customer_id."""
    mock_resp = MagicMock()
    mock_resp.data = [{"customer_id": "CUST-001", "features": {"tenure": 5, "MonthlyCharges": 70.0}, "churn_label": False}]
    mock_supabase_client.table().select().eq().limit().execute.return_value = mock_resp

    features = get_customer_features("CUST-001", client=mock_supabase_client)
    assert features is not None
    assert features["customer_id"] == "CUST-001"
    assert features["tenure"] == 5


def test_get_customer_features_not_found(mock_supabase_client):
    """Verifies None returned when customer does not exist."""
    mock_resp = MagicMock()
    mock_resp.data = []
    mock_supabase_client.table().select().eq().limit().execute.return_value = mock_resp

    features = get_customer_features("NON-EXISTENT", client=mock_supabase_client)
    assert features is None


def test_insert_prediction_mock(mock_supabase_client):
    """Verifies prediction persistence with probability and model version."""
    mock_resp = MagicMock()
    mock_resp.data = [{"id": 1, "customer_id": "CUST-001", "churn_probability": 0.85, "model_version": "v1.0.0"}]
    mock_supabase_client.table().insert().execute.return_value = mock_resp

    res = insert_prediction("CUST-001", 0.85, "v1.0.0", client=mock_supabase_client)
    assert res["customer_id"] == "CUST-001"
    assert res["churn_probability"] == 0.85
    mock_supabase_client.table.assert_called_with("predictions")


def test_insert_model_run_mock(mock_supabase_client):
    """Verifies model run logging to Supabase."""
    mock_resp = MagicMock()
    mock_resp.data = [{"id": 10, "run_id": "mlflow-run-123", "model_name": "XGBoost", "roc_auc": 0.852, "f1": 0.62}]
    mock_supabase_client.table().upsert().execute.return_value = mock_resp

    res = insert_model_run("mlflow-run-123", "XGBoost", 0.852, 0.62, client=mock_supabase_client)
    assert res["run_id"] == "mlflow-run-123"
    assert res["roc_auc"] == 0.852
