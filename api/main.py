"""FastAPI REST API serving customer churn prediction models.

Endpoints:
    - GET  /health: Health check and model readiness status.
    - POST /predict: Predicts churn probability for raw customer JSON features.
    - GET  /predict/{customer_id}: Predicts churn for a customer stored in Supabase.
    - GET  /model-info: Returns the champion model name, metrics, and metadata.
"""

from pathlib import Path
import sys
import json
from typing import Any, Dict, Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    API_HOST,
    API_PORT,
    BEST_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
    MODEL_METADATA_PATH,
    setup_logger,
)
from src.predict import get_model, predict_customer, predict_features

logger = setup_logger("api.main")


class CustomerFeaturesInput(BaseModel):
    """Pydantic schema validating input features for churn inference."""

    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(0, ge=0, le=1, description="Whether customer is a senior citizen (1 or 0)")
    Partner: Literal["Yes", "No"] = Field("No", description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field("No", description="Whether customer has dependents")
    tenure: float = Field(..., ge=0, description="Months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field("Yes", description="Whether customer has phone service")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field("No", description="Multiple lines subscription")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field("Fiber optic", description="Internet service provider")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field("No", description="Online security add-on")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field("No", description="Online backup add-on")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field("No", description="Device protection add-on")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field("No", description="Tech support add-on")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field("No", description="Streaming TV add-on")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field("No", description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        "Month-to-month", description="Contract duration"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field("Yes", description="Paperless billing preference")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field("Electronic check", description="Billing payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges amount in dollars")
    TotalCharges: Optional[float] = Field(None, ge=0.0, description="Total charges cumulative in dollars")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "TotalCharges": 239.55,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response payload."""

    churn_probability: float
    risk_label: Literal["low", "medium", "high"]
    model_version: str
    customer_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for service health check response."""

    status: str
    model_loaded: bool
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Initializing API application...")
    if BEST_MODEL_PATH.exists():
        try:
            get_model()
            logger.info("Preloaded best model on startup.")
        except Exception as exc:
            logger.warning("Failed to preload model at startup: %s", exc)
    else:
        logger.warning("Model file not found at %s. API will start without warm model.", BEST_MODEL_PATH)
    yield
    logger.info("Shutting down API application...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API for real-time customer churn scoring and Supabase integration.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit dashboard and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Performs system liveness and model availability check."""
    model_loaded = False
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    version = DEFAULT_MODEL_VERSION
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                version = meta.get("version", DEFAULT_MODEL_VERSION)
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version=version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: CustomerFeaturesInput) -> PredictionResponse:
    """Computes churn probability and risk tier for a customer feature payload."""
    try:
        model = get_model()
    except FileNotFoundError as fnf_err:
        logger.error("Prediction requested but model is missing: %s", fnf_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not available. Please run model training first.",
        )
    except Exception as exc:
        logger.error("Error accessing model: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while loading model pipeline.",
        )

    try:
        features_dict = payload.model_dump()
        # Default TotalCharges if omitted
        if features_dict.get("TotalCharges") is None:
            features_dict["TotalCharges"] = features_dict["tenure"] * features_dict["MonthlyCharges"]

        prob, risk = predict_features(features_dict, model=model)

        version = DEFAULT_MODEL_VERSION
        if MODEL_METADATA_PATH.exists():
            try:
                with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    version = json.load(f).get("version", DEFAULT_MODEL_VERSION)
            except Exception:
                pass

        return PredictionResponse(
            churn_probability=round(prob, 4),
            risk_label=risk,
            model_version=version,
        )
    except Exception as exc:
        logger.error("Prediction failed for feature payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.get("/predict/{customer_id}", response_model=PredictionResponse, tags=["Inference"])
def predict_by_customer_id(customer_id: str) -> PredictionResponse:
    """Fetches customer features from Supabase, runs inference, and logs result."""
    try:
        result = predict_customer(customer_id=customer_id, record_prediction=True)
        return PredictionResponse(
            customer_id=result["customer_id"],
            churn_probability=result["churn_probability"],
            risk_label=result["risk_label"],
            model_version=result["model_version"],
        )
    except ValueError as ve:
        logger.warning("Customer lookup failed: %s", ve)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )
    except FileNotFoundError as fnf:
        logger.error("Model file missing: %s", fnf)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained yet.",
        )
    except Exception as exc:
        logger.error("Error processing predict for customer %s: %s", customer_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing customer prediction: {str(exc)}",
        )


@app.get("/model-info", tags=["Metadata"])
def get_model_info() -> Dict[str, Any]:
    """Returns active champion model name, hyperparameters, evaluation metrics, and metadata."""
    if not MODEL_METADATA_PATH.exists():
        return {
            "model_name": "No model metadata recorded",
            "version": DEFAULT_MODEL_VERSION,
            "status": "untrained",
            "metrics": {},
        }

    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except Exception as exc:
        logger.error("Failed to read model metadata: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading model metadata file.",
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting API server on %s:%d", API_HOST, API_PORT)
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
