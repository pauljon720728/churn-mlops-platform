"""Prediction and inference service for churn-mlops-platform.

Loads the champion model pipeline, generates churn risk probabilities
and risk tiers for incoming feature vectors or Supabase customer records,
and writes predictions into the Supabase 'predictions' table.
"""

from pathlib import Path
import sys
import json
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import pandas as pd

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
    MODEL_METADATA_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
    setup_logger,
)
from src.db import get_customer_features, insert_prediction

logger = setup_logger(__name__)

_loaded_model: Optional[Any] = None
_model_version: str = DEFAULT_MODEL_VERSION


def get_model(model_path: Optional[Path] = None) -> Any:
    """Returns the cached champion model pipeline or loads it from disk.

    Args:
        model_path: Optional explicit model path.

    Returns:
        Any: Serialized imblearn/scikit-learn pipeline.

    Raises:
        FileNotFoundError: If the model file is not present.
    """
    global _loaded_model, _model_version
    if _loaded_model is not None:
        return _loaded_model

    target_path = model_path or BEST_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {target_path}. Please execute train.py first."
        )

    logger.info("Loading model artifact from %s", target_path)
    _loaded_model = joblib.load(target_path)

    # Read version from metadata if available
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                _model_version = meta.get("version", DEFAULT_MODEL_VERSION)
        except Exception as exc:
            logger.warning("Could not read model metadata: %s", exc)

    logger.info("Model loaded successfully (version: %s)", _model_version)
    return _loaded_model


def get_risk_label(probability: float) -> str:
    """Categorizes churn probability into risk tiers.

    Args:
        probability: Float between 0.0 and 1.0.

    Returns:
        str: 'low', 'medium', or 'high'.
    """
    if probability < RISK_THRESHOLD_LOW:
        return "low"
    elif probability <= RISK_THRESHOLD_HIGH:
        return "medium"
    return "high"


def predict_features(
    features: Union[Dict[str, Any], pd.DataFrame],
    model: Optional[Any] = None,
) -> Tuple[float, str]:
    """Generates churn probability and risk tier directly from raw customer features.

    Args:
        features: Dictionary or DataFrame representing customer attributes.
        model: Optional pre-loaded model pipeline.

    Returns:
        Tuple[float, str]: (churn_probability, risk_label)
    """
    pipeline = model or get_model()

    if isinstance(features, dict):
        df_input = pd.DataFrame([features])
    else:
        df_input = features.copy()

    # Drop ID and target columns if present in features
    for col in ["customerID", "customer_id", "Churn", "churn_label"]:
        if col in df_input.columns:
            df_input = df_input.drop(columns=[col])

    # Compute probability of positive class (churn)
    prob_array = pipeline.predict_proba(df_input)
    churn_prob = float(prob_array[0, 1])
    risk_label = get_risk_label(churn_prob)

    return churn_prob, risk_label


def predict_customer(
    customer_id: str,
    record_prediction: bool = True,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retrieves customer features from Supabase, runs inference, and logs to predictions table.

    Args:
        customer_id: Unique customer ID.
        record_prediction: If True, writes the result to Supabase predictions table.
        client: Optional Supabase client instance.

    Returns:
        Dict[str, Any]: Prediction payload containing probability, risk label, and metadata.

    Raises:
        ValueError: If the customer_id is not found in the database.
    """
    logger.info("Handling prediction request for customer_id: %s", customer_id)
    customer_data = get_customer_features(customer_id, client=client)

    if not customer_data:
        raise ValueError(f"Customer with ID '{customer_id}' was not found in Supabase.")

    # Extract features dictionary
    raw_features = customer_data.get("features", customer_data)
    churn_prob, risk_label = predict_features(raw_features)

    # Persist inference record
    if record_prediction:
        try:
            insert_prediction(
                customer_id=customer_id,
                churn_probability=churn_prob,
                model_version=_model_version,
                client=client,
            )
        except Exception as exc:
            logger.warning("Could not persist prediction to Supabase for customer %s: %s", customer_id, exc)

    return {
        "customer_id": customer_id,
        "churn_probability": round(churn_prob, 4),
        "risk_label": risk_label,
        "model_version": _model_version,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cid = sys.argv[1]
        try:
            res = predict_customer(cid)
            logger.info("Result for %s: %s", cid, res)
        except Exception as e:
            logger.error("Prediction failed: %s", e)
    else:
        logger.info("Usage: python src/predict.py <customer_id>")
