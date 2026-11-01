"""Unit tests for feature engineering, cleaning, and preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    TelcoFeatureEngineer,
    build_preprocessing_pipeline,
    get_all_feature_columns,
    get_transformed_feature_names,
)
from src.ingest import clean_raw_data, prepare_supabase_payload


def test_clean_raw_data_handles_total_charges_and_churn(sample_raw_dataframe):
    """Verifies whitespace TotalCharges conversion and Churn boolean mapping."""
    cleaned = clean_raw_data(sample_raw_dataframe)

    # TotalCharges must be float, no NaNs
    assert np.issubdtype(cleaned["TotalCharges"].dtype, np.floating)
    assert cleaned["TotalCharges"].isna().sum() == 0
    # Customer 0002-TESTB had whitespace ' ' and tenure 0 -> should be 0.0
    assert cleaned.loc[cleaned["customerID"] == "0002-TESTB", "TotalCharges"].values[0] == 0.0

    # Churn must be boolean
    assert cleaned["Churn"].dtype == bool
    assert bool(cleaned.loc[cleaned["customerID"] == "0001-TESTA", "Churn"].values[0]) is False
    assert bool(cleaned.loc[cleaned["customerID"] == "0002-TESTB", "Churn"].values[0]) is True


def test_clean_raw_data_drops_duplicates():
    """Verifies duplicate customerIDs are removed."""
    df_with_dups = pd.DataFrame(
        [
            {"customerID": "DUP-1", "TotalCharges": "100.0", "Churn": "No"},
            {"customerID": "DUP-1", "TotalCharges": "100.0", "Churn": "No"},
            {"customerID": "DUP-2", "TotalCharges": "200.0", "Churn": "Yes"},
        ]
    )
    cleaned = clean_raw_data(df_with_dups)
    assert len(cleaned) == 2
    assert list(cleaned["customerID"]) == ["DUP-1", "DUP-2"]


def test_feature_engineer_transformer(sample_customer_dict):
    """Verifies TelcoFeatureEngineer generates expected interaction features."""
    engineer = TelcoFeatureEngineer()
    res_df = engineer.transform(sample_customer_dict)

    assert "tenure_bucket" in res_df.columns
    assert "charges_per_tenure" in res_df.columns
    assert "num_services" in res_df.columns
    assert "monthly_to_total_ratio" in res_df.columns
    assert "has_streaming" in res_df.columns
    assert "has_security_backup" in res_df.columns

    # Check calculated values
    assert res_df["tenure_bucket"].values[0] == "0-12m"
    # Services: PhoneService=Yes, OnlineBackup=Yes, StreamingTV=Yes, StreamingMovies=Yes -> total 4
    assert res_df["num_services"].values[0] == 4
    assert res_df["has_streaming"].values[0] == "Yes"
    assert res_df["has_security_backup"].values[0] == "Yes"


def test_preprocessing_pipeline_fit_transform(sample_raw_dataframe):
    """Verifies full preprocessing pipeline transforms data without leakage or NaNs."""
    df = clean_raw_data(sample_raw_dataframe)
    X = df.drop(columns=["customerID", "Churn"])

    pipeline = build_preprocessing_pipeline()
    X_trans = pipeline.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(df)
    assert not np.isnan(X_trans).any(), "Transformed output must not contain NaN values"

    feature_names = get_transformed_feature_names(pipeline)
    assert len(feature_names) == X_trans.shape[1]
    assert any("tenure" in name for name in feature_names)


def test_prepare_supabase_payload(sample_raw_dataframe):
    """Verifies JSONB structure generated for Supabase upsert."""
    cleaned = clean_raw_data(sample_raw_dataframe)
    payloads = prepare_supabase_payload(cleaned)

    assert len(payloads) == len(cleaned)
    first = payloads[0]
    assert "customer_id" in first
    assert "features" in first
    assert "churn_label" in first
    assert isinstance(first["features"], dict)
    assert "customerID" not in first["features"]
    assert "Churn" not in first["features"]
