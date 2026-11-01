"""Supabase database client and data access helpers.

Provides typed interfaces for upserting customer records, querying customer
features, recording prediction inferences, and logging model experiment runs.
"""

from typing import Any, Dict, List, Optional
from supabase import create_client, Client
from src.config import SUPABASE_KEY, SUPABASE_URL, setup_logger

logger = setup_logger(__name__)

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Initializes and returns the singleton Supabase client.

    Returns:
        Client: Authenticated Supabase client instance.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY are invalid or missing.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY or "dummy" in SUPABASE_URL:
        logger.warning(
            "SUPABASE_URL or SUPABASE_KEY is unset or using dummy placeholder. "
            "Database operations requiring remote access will fail."
        )

    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully against %s", SUPABASE_URL)
        return _supabase_client
    except Exception as exc:
        logger.error("Failed to initialize Supabase client: %s", exc)
        raise


def upsert_customers(records: List[Dict[str, Any]], client: Optional[Client] = None) -> int:
    """Upserts a batch of customer records into the customers table.

    Uses `customer_id` as the conflict target.

    Args:
        records: List of dictionaries matching the customers table schema:
                 [{"customer_id": str, "features": dict, "churn_label": bool}]
        client: Optional Supabase client instance for dependency injection/testing.

    Returns:
        int: Number of records successfully processed.
    """
    if not records:
        logger.info("No records provided to upsert_customers.")
        return 0

    supabase = client or get_supabase_client()
    try:
        response = supabase.table("customers").upsert(
            records,
            on_conflict="customer_id"
        ).execute()

        count = len(response.data) if response.data else len(records)
        logger.info("Successfully upserted %d records into 'customers' table.", count)
        return count
    except Exception as exc:
        logger.error("Error upserting customer batch of size %d: %s", len(records), exc)
        raise


def get_customer_features(customer_id: str, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Retrieves customer features JSON from the customers table.

    Args:
        customer_id: Unique customer ID string.
        client: Optional Supabase client instance.

    Returns:
        Optional[Dict[str, Any]]: Customer feature dictionary if found, else None.
    """
    supabase = client or get_supabase_client()
    try:
        response = (
            supabase.table("customers")
            .select("customer_id, features, churn_label")
            .eq("customer_id", customer_id)
            .limit(1)
            .execute()
        )
        if response.data and len(response.data) > 0:
            row = response.data[0]
            logger.info("Retrieved customer record for customer_id: %s", customer_id)
            features = row.get("features", {})
            features["customer_id"] = row.get("customer_id")
            if "churn_label" in row:
                features["churn_label"] = row.get("churn_label")
            return features

        logger.warning("No customer found matching customer_id: %s", customer_id)
        return None
    except Exception as exc:
        logger.error("Error retrieving features for customer_id '%s': %s", customer_id, exc)
        raise


def insert_prediction(
    customer_id: str,
    churn_probability: float,
    model_version: str,
    client: Optional[Client] = None,
) -> Dict[str, Any]:
    """Records a single inference prediction into the predictions table.

    Args:
        customer_id: Unique customer identifier.
        churn_probability: Predicted probability of churn (0.0 to 1.0).
        model_version: Model identifier or version string.
        client: Optional Supabase client instance.

    Returns:
        Dict[str, Any]: The inserted record.
    """
    supabase = client or get_supabase_client()
    payload = {
        "customer_id": customer_id,
        "churn_probability": float(churn_probability),
        "model_version": model_version,
    }
    try:
        response = supabase.table("predictions").insert(payload).execute()
        logger.info(
            "Logged prediction for %s (prob=%.4f, version=%s)",
            customer_id,
            churn_probability,
            model_version,
        )
        return response.data[0] if response.data else payload
    except Exception as exc:
        logger.error("Failed to insert prediction for customer %s: %s", customer_id, exc)
        raise


def insert_model_run(
    run_id: str,
    model_name: str,
    roc_auc: float,
    f1: float,
    client: Optional[Client] = None,
) -> Dict[str, Any]:
    """Records experiment model run metrics into the model_runs table.

    Args:
        run_id: Unique MLflow or generated run UUID.
        model_name: Name of the trained architecture.
        roc_auc: Evaluation ROC-AUC score.
        f1: Evaluation F1 score.
        client: Optional Supabase client instance.

    Returns:
        Dict[str, Any]: The inserted model run record.
    """
    supabase = client or get_supabase_client()
    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "roc_auc": float(roc_auc),
        "f1": float(f1),
    }
    try:
        response = supabase.table("model_runs").upsert(
            payload,
            on_conflict="run_id"
        ).execute()
        logger.info("Logged model run '%s' (%s, ROC-AUC=%.4f, F1=%.4f)", run_id, model_name, roc_auc, f1)
        return response.data[0] if response.data else payload
    except Exception as exc:
        logger.error("Failed to insert model run record %s: %s", run_id, exc)
        raise


def get_recent_predictions(
    threshold: Optional[float] = None,
    limit: int = 100,
    client: Optional[Client] = None,
) -> List[Dict[str, Any]]:
    """Fetches recent predictions, optionally filtered by minimum churn probability.

    Args:
        threshold: Minimum churn probability threshold filter.
        limit: Maximum number of rows to retrieve.
        client: Optional Supabase client instance.

    Returns:
        List[Dict[str, Any]]: List of prediction records.
    """
    supabase = client or get_supabase_client()
    try:
        query = supabase.table("predictions").select("*").order("predicted_at", desc=True).limit(limit)
        if threshold is not None:
            query = query.gte("churn_probability", threshold)
        response = query.execute()
        return response.data or []
    except Exception as exc:
        logger.error("Failed to query recent predictions: %s", exc)
        return []


def get_latest_model_run(client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent model run record from Supabase.

    Args:
        client: Optional Supabase client instance.

    Returns:
        Optional[Dict[str, Any]]: Recent model run record if found.
    """
    supabase = client or get_supabase_client()
    try:
        response = (
            supabase.table("model_runs")
            .select("*")
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as exc:
        logger.error("Failed to query latest model run: %s", exc)
        return None
