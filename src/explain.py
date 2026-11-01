"""Model explainability module using SHAP for churn-mlops-platform.

Computes SHAP (SHapley Additive exPlanations) values for the best trained model,
generates feature importance summary and bar visualizations, persists plots to
the models directory for dashboard display, and logs artifacts to MLflow.
"""

from pathlib import Path
import sys
import argparse
from typing import List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    ID_COLUMN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
    SHAP_BAR_PATH,
    SHAP_SUMMARY_PATH,
    TARGET_COLUMN,
    setup_logger,
)
from src.features import get_transformed_feature_names
from src.ingest import clean_raw_data

logger = setup_logger(__name__)


def generate_shap_explanations(
    model_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    sample_size: int = 400,
) -> Tuple[Path, Path]:
    """Generates SHAP summary and feature importance bar plots for the best model.

    Args:
        model_path: Path to serialized joblib pipeline. Defaults to BEST_MODEL_PATH.
        csv_path: Path to dataset CSV for background distribution. Defaults to RAW_DATA_PATH.
        sample_size: Number of background/test instances to sample for tractable SHAP compute.

    Returns:
        Tuple[Path, Path]: (summary_plot_path, bar_plot_path)
    """
    m_path = model_path or BEST_MODEL_PATH
    d_path = csv_path or RAW_DATA_PATH

    if not m_path.exists():
        raise FileNotFoundError(f"Serialized model not found at {m_path}. Run training first.")
    if not d_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {d_path}.")

    logger.info("Loading model pipeline from %s", m_path)
    pipeline = joblib.load(m_path)

    logger.info("Loading dataset from %s", d_path)
    raw_df = pd.read_csv(d_path)
    cleaned_df = clean_raw_data(raw_df)

    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in cleaned_df.columns]
    X_raw = cleaned_df.drop(columns=drop_cols)

    # Subsample for efficient SHAP explanation computation
    if len(X_raw) > sample_size:
        X_sample = X_raw.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_raw

    # Extract preprocessing pipeline and classifier step
    if "engineer" in pipeline.named_steps:
        X_sample = pipeline.named_steps["engineer"].transform(X_sample)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    logger.info("Transforming evaluation samples through preprocessing pipeline...")
    X_trans = preprocessor.transform(X_sample)
    feature_names = get_transformed_feature_names(preprocessor)

    # If feature_names count doesn't match columns, use generic fallback names
    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    logger.info("Initializing SHAP Explainer for classifier: %s", type(classifier).__name__)
    try:
        # Tree-based explainer for XGBoost and Random Forest
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_trans)
    except Exception as exc:
        logger.info("TreeExplainer not directly applicable (%s); falling back to general Explainer.", exc)
        explainer = shap.Explainer(classifier.predict_proba, X_trans)
        shap_explanation = explainer(X_trans)
        # Select positive class (churn=1)
        shap_values = shap_explanation.values[:, :, 1] if len(shap_explanation.values.shape) == 3 else shap_explanation.values

    # For binary classification with TreeExplainer, shap_values may be a list of [neg, pos] or 3D array
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals_positive = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_vals_positive = shap_values[:, :, 1]
    else:
        shap_vals_positive = shap_values

    # 1. SHAP Beeswarm Summary Plot
    logger.info("Generating SHAP beeswarm summary plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals_positive,
        X_trans,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Feature Importance & Impact (Beeswarm)", fontsize=14, pad=15)
    plt.tight_layout()
    SHAP_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SHAP_SUMMARY_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", SHAP_SUMMARY_PATH)

    # 2. SHAP Bar Plot (Global Importance)
    logger.info("Generating SHAP global importance bar plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_positive,
        X_trans,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP bar plot to %s", SHAP_BAR_PATH)

    # Log to MLflow if tracking active
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="shap_explainability"):
            mlflow.log_artifact(str(SHAP_SUMMARY_PATH), artifact_path="explainability")
            mlflow.log_artifact(str(SHAP_BAR_PATH), artifact_path="explainability")
            logger.info("Logged SHAP plots to MLflow under 'explainability' artifact path.")
    except Exception as exc:
        logger.warning("Could not log SHAP artifacts to MLflow: %s", exc)

    return SHAP_SUMMARY_PATH, SHAP_BAR_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explainability plots.")
    parser.add_argument("--model-path", type=str, default=str(BEST_MODEL_PATH), help="Path to best model")
    parser.add_argument("--csv-path", type=str, default=str(RAW_DATA_PATH), help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of instances to explain")

    args = parser.parse_args()
    generate_shap_explanations(
        model_path=Path(args.model_path),
        csv_path=Path(args.csv_path),
        sample_size=args.sample_size,
    )
