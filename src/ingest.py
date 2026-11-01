"""Data ingestion pipeline for IBM Telco Customer Churn dataset.

Cleans raw CSV data, handles type conversions, normalizes TotalCharges,
converts the target Churn label to boolean, and upserts customer records
into Supabase in batches of 500 rows.
"""

from pathlib import Path
import os
import sys
import argparse
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    ID_COLUMN,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    setup_logger,
)
from src.db import upsert_customers

logger = setup_logger(__name__)

BATCH_SIZE: int = 500


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs essential cleaning and validation on raw Telco churn dataframe.

    Operations:
        1. Drops exact duplicate rows and duplicate customer IDs.
        2. Converts TotalCharges to numeric, replacing blank strings with 0.0
           (which correspond to newly acquired customers where tenure is 0).
        3. Maps Churn 'Yes'/'No' strings to boolean True/False.

    Args:
        df: Raw pandas DataFrame loaded from CSV.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_count = len(df)
    logger.info("Starting raw data cleaning with %d records.", initial_count)

    # 1. Drop duplicate customer IDs
    df_clean = df.drop_duplicates(subset=[ID_COLUMN], keep="first").copy()
    duplicates_removed = initial_count - len(df_clean)
    if duplicates_removed > 0:
        logger.warning("Removed %d duplicate customer records.", duplicates_removed)

    # 2. Fix TotalCharges (blank strings to numeric float)
    if "TotalCharges" in df_clean.columns:
        # Replace empty strings or pure spaces with NaN, then convert
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(),
            errors="coerce",
        )
        # When tenure is 0, TotalCharges is legitimately 0.0
        missing_total = df_clean["TotalCharges"].isna().sum()
        if missing_total > 0:
            logger.info("Found %d records with empty TotalCharges; imputing with 0.0 (tenure=0).", missing_total)
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # 3. Standardize and convert Churn to boolean
    if TARGET_COLUMN in df_clean.columns:
        churn_mapped = df_clean[TARGET_COLUMN].astype(str).str.strip().str.lower()
        df_clean[TARGET_COLUMN] = churn_mapped.map({"yes": True, "no": False, "1": True, "0": False})
        logger.info(
            "Churn distribution: %s",
            df_clean[TARGET_COLUMN].value_counts().to_dict(),
        )

    logger.info("Cleaned dataframe contains %d rows and %d columns.", len(df_clean), df_clean.shape[1])
    return df_clean


def prepare_supabase_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts cleaned DataFrame into JSONB payload matching Supabase schema.

    Schema:
        id: bigserial (auto)
        customer_id: text unique not null
        features: jsonb not null
        churn_label: boolean

    Args:
        df: Cleaned DataFrame.

    Returns:
        List[Dict[str, Any]]: List of dictionary payloads ready for Supabase upsert.
    """
    payloads: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        customer_id = str(row[ID_COLUMN])
        churn_label = bool(row[TARGET_COLUMN]) if TARGET_COLUMN in row and pd.notna(row[TARGET_COLUMN]) else None

        # Feature dictionary excludes customerID and Churn label
        features: Dict[str, Any] = {}
        for col in df.columns:
            if col in (ID_COLUMN, TARGET_COLUMN):
                continue
            val = row[col]
            if pd.isna(val):
                features[col] = None
            elif isinstance(val, (np.integer, int)):
                features[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                features[col] = float(val)
            elif isinstance(val, (np.bool_, bool)):
                features[col] = bool(val)
            else:
                features[col] = str(val)

        payloads.append(
            {
                "customer_id": customer_id,
                "features": features,
                "churn_label": churn_label,
            }
        )

    return payloads


def run_ingest(
    csv_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
) -> int:
    """Executes the complete ingestion pipeline.

    Args:
        csv_path: Path to raw CSV file. Defaults to RAW_DATA_PATH.
        batch_size: Batch size for database upserts (default: 500).
        dry_run: If True, executes cleaning and batch partitioning without sending to Supabase.

    Returns:
        int: Total number of records processed.
    """
    path = csv_path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Source CSV dataset not found at: {path}")

    logger.info("Reading raw dataset from %s", path)
    raw_df = pd.read_csv(path)
    logger.info("Loaded %d rows from CSV.", len(raw_df))

    cleaned_df = clean_raw_data(raw_df)
    records = prepare_supabase_payload(cleaned_df)
    total_records = len(records)
    logger.info("Generated %d Supabase customer payloads.", total_records)

    if dry_run:
        logger.info("Dry run requested. Skipping remote Supabase upsert for %d records.", total_records)
        return total_records

    total_upserted = 0
    total_batches = (total_records + batch_size - 1) // batch_size
    logger.info("Beginning batch upsert (%d batches of size %d).", total_batches, batch_size)

    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        batch_idx = (i // batch_size) + 1
        logger.info("Processing batch %d/%d (%d rows)...", batch_idx, total_batches, len(batch))
        try:
            upsert_customers(batch)
            total_upserted += len(batch)
        except Exception as exc:
            logger.error("Failed to upsert batch %d/%d: %s", batch_idx, total_batches, exc)
            raise

    logger.info("Data ingestion completed. Total records upserted: %d", total_upserted)
    return total_upserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Telco Churn CSV data into Supabase.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Path to source CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for database upserts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clean and validate data without writing to Supabase",
    )

    args = parser.parse_args()
    run_ingest(
        csv_path=Path(args.csv_path),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
