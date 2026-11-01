"""Integration and endpoint tests for FastAPI customer churn service."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client(mock_trained_pipeline):
    """Provides a TestClient with pre-injected champion model."""
    with patch("api.main.get_model", return_value=mock_trained_pipeline), \
         patch("src.predict.get_model", return_value=mock_trained_pipeline):
        yield TestClient(app)


def test_health_endpoint(client):
    """Verifies /health endpoint returns 200 status and readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "version" in data


def test_predict_endpoint_valid_payload(client, sample_customer_dict):
    """Verifies POST /predict successfully generates probability and risk tier."""
    response = client.post("/predict", json=sample_customer_dict)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_label"] in ["low", "medium", "high"]
    assert "model_version" in data


def test_predict_endpoint_invalid_payload(client):
    """Verifies POST /predict returns 422 Unprocessable Entity for schema violations."""
    invalid_payload = {
        "gender": "UnknownGender",  # invalid literal
        "tenure": -5,  # must be ge 0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_by_customer_id_success(client, sample_customer_dict):
    """Verifies GET /predict/{customer_id} fetches from DB and returns inference."""
    mock_customer_record = {
        "customer_id": "7590-VHVEG",
        "features": sample_customer_dict,
        "churn_label": True,
    }

    with patch("src.predict.get_customer_features", return_value=mock_customer_record), \
         patch("src.predict.insert_prediction", return_value={}):
        response = client.get("/predict/7590-VHVEG")
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "7590-VHVEG"
        assert "churn_probability" in data
        assert data["risk_label"] in ["low", "medium", "high"]


def test_predict_by_customer_id_not_found(client):
    """Verifies GET /predict/{customer_id} returns 404 when customer ID doesn't exist."""
    with patch("src.predict.get_customer_features", return_value=None):
        response = client.get("/predict/UNKNOWN-ID")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_model_info_endpoint(client):
    """Verifies GET /model-info returns configuration and metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data or "version" in data
