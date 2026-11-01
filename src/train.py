"""Model training and evaluation pipeline for churn-mlops-platform.

Trains and evaluates multiple candidate models:
    1. Logistic Regression (Baseline with class weighting)
    2. Random Forest (with SMOTE and class weighting)
    3. XGBoost Classifier (with SMOTE / hyperparameter optimization)

Evaluates candidates using stratified cross-validation on ROC-AUC, F1,
precision, recall, and confusion matrix. Logs all runs, metrics, and
plots to MLflow, records the winning run into Supabase 'model_runs', and
serializes the best pipeline to models/best_model.joblib.
"""

from pathlib import Path
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import mlflow
import mlflow.sklearn

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
    ID_COLUMN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    setup_logger,
)
from src.features import build_preprocessing_pipeline, get_preprocessing_steps
from src.ingest import clean_raw_data
from src.db import insert_model_run

logger = setup_logger(__name__)


def load_and_split_data(
    csv_path: Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads raw CSV data, executes cleaning, and creates stratified train/test partitions.

    Args:
        csv_path: Filepath to source CSV dataset.
        test_size: Proportion of dataset allocated to the holdout test set.
        random_state: Reproducibility seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading dataset for training from %s", csv_path)
    raw_df = pd.read_csv(csv_path)
    clean_df = clean_raw_data(raw_df)

    # Features: drop ID column and target
    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in clean_df.columns]
    X = clean_df.drop(columns=drop_cols)
    y = clean_df[TARGET_COLUMN].astype(int)

    logger.info("Splitting dataset into train (%.0f%%) and test (%.0f%%) using Stratified split.", (1 - test_size) * 100, test_size * 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Train set: %d samples (churn=%.2f%%), Test set: %d samples (churn=%.2f%%)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def plot_and_save_artifacts(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generates and saves Confusion Matrix and ROC Curve visualization artifacts.

    Args:
        y_true: Ground truth test binary labels.
        y_pred: Predicted class labels.
        y_prob: Predicted positive class probabilities.
        model_name: Model identifier for title/filename.
        output_dir: Directory where PNG plots are saved.

    Returns:
        Tuple[Path, Path]: (cm_plot_path, roc_plot_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(" ", "_")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = output_dir / f"{slug}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    auc_val = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = output_dir / f"{slug}_roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close(fig)

    return cm_path, roc_path


def train_and_evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    param_grid: Optional[Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 8,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Trains a model pipeline with optional hyperparameter tuning and evaluates metrics.

    Args:
        model_name: Descriptive name of the model.
        pipeline: imblearn Pipeline instance.
        param_grid: Parameter distribution for RandomizedSearchCV if tuning.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        n_iter: Number of parameter settings sampled in RandomizedSearchCV.
        cv_folds: Number of stratified cross-validation folds.
        random_state: Reproducibility seed.

    Returns:
        Dict[str, Any]: Model results dictionary with metrics and fitted estimator.
    """
    logger.info("Training candidate: %s ...", model_name)
    stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if param_grid:
        logger.info("Running RandomizedSearchCV tuning (%d iterations, %d folds)...", n_iter, cv_folds)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=stratified_cv,
            random_state=random_state,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
        logger.info("Best CV ROC-AUC: %.4f | Best params: %s", cv_score, best_params)
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = {}
        cv_score = 0.0

    # Predictions on holdout test set
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "cv_roc_auc": float(cv_score),
    }

    logger.info(
        "Evaluation results for %s -> ROC-AUC: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f",
        model_name,
        metrics["roc_auc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )

    return {
        "name": model_name,
        "pipeline": best_pipeline,
        "params": best_params,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def run_training_suite(
    csv_path: Optional[Path] = None,
    tune_hyperparameters: bool = True,
    tuning_iterations: int = 8,
) -> Dict[str, Any]:
    """Orchestrates candidate model training, MLflow experiment tracking, and artifact persistence.

    Args:
        csv_path: Path to dataset CSV. Defaults to RAW_DATA_PATH.
        tune_hyperparameters: If True, executes RandomizedSearchCV tuning on ensemble models.
        tuning_iterations: Number of parameter combinations to sample during tuning.

    Returns:
        Dict[str, Any]: Summary of winning model and experiment results.
    """
    path = csv_path or RAW_DATA_PATH
    X_train, X_test, y_train, y_test = load_and_split_data(path)

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow configured at '%s' for experiment '%s'", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    prep_steps = get_preprocessing_steps()

    candidates = [
        # 1. Baseline: Logistic Regression with balanced class weights
        {
            "name": "Logistic Regression Baseline",
            "pipeline": ImbPipeline(
                steps=[
                    prep_steps[0],
                    prep_steps[1],
                    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
                ]
            ),
            "param_grid": {
                "classifier__C": np.logspace(-2, 2, 20),
                "classifier__solver": ["lbfgs", "liblinear"],
            } if tune_hyperparameters else None,
        },
        # 2. Random Forest with SMOTE and balanced class weights
        {
            "name": "Random Forest with SMOTE",
            "pipeline": ImbPipeline(
                steps=[
                    prep_steps[0],
                    prep_steps[1],
                    ("sampler", SMOTE(random_state=42)),
                    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42)),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [4, 6, 8, 12, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            } if tune_hyperparameters else None,
        },
        # 3. XGBoost Classifier with cost-sensitive class weighting
        {
            "name": "XGBoost Classifier",
            "pipeline": ImbPipeline(
                steps=[
                    prep_steps[0],
                    prep_steps[1],
                    (
                        "classifier",
                        xgb.XGBClassifier(
                            eval_metric="logloss",
                            random_state=42,
                            scale_pos_weight=float((y_train == 0).sum() / max(1, (y_train == 1).sum())),
                            n_estimators=120,
                            max_depth=4,
                            learning_rate=0.04,
                            subsample=0.85,
                            colsample_bytree=0.85,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [3, 4, 5],
                "classifier__learning_rate": [0.03, 0.05, 0.08],
                "classifier__subsample": [0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            } if tune_hyperparameters else None,
        },
    ]

    results = []
    best_candidate = None
    best_score = -1.0

    for candidate_def in candidates:
        model_name = candidate_def["name"]
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            logger.info("Started MLflow run %s for '%s'", run_id, model_name)

            res = train_and_evaluate_model(
                model_name=model_name,
                pipeline=candidate_def["pipeline"],
                param_grid=candidate_def["param_grid"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=tuning_iterations,
            )
            res["run_id"] = run_id

            # Log parameters to MLflow
            if res["params"]:
                clean_params = {k.replace("classifier__", ""): str(v) for k, v in res["params"].items()}
                mlflow.log_params(clean_params)

            # Log metrics to MLflow
            mlflow.log_metrics(res["metrics"])

            # Generate and log visualization artifacts
            cm_plot, roc_plot = plot_and_save_artifacts(
                y_true=y_test,
                y_pred=res["y_pred"],
                y_prob=res["y_prob"],
                model_name=model_name,
                output_dir=MODELS_DIR,
            )
            mlflow.log_artifact(str(cm_plot), artifact_path="plots")
            mlflow.log_artifact(str(roc_plot), artifact_path="plots")

            # Log model to MLflow
            try:
                mlflow.sklearn.log_model(
                    sk_model=res["pipeline"],
                    artifact_path="model",
                    serialization_format="cloudpickle",
                )
            except Exception as log_err:
                logger.warning("Could not log serialized model to MLflow: %s", log_err)

            # Record model run to Supabase
            try:
                insert_model_run(
                    run_id=run_id,
                    model_name=model_name,
                    roc_auc=res["metrics"]["roc_auc"],
                    f1=res["metrics"]["f1"],
                )
            except Exception as exc:
                logger.warning("Could not persist model run %s to Supabase: %s", run_id, exc)

            results.append(res)

            # Rank by ROC-AUC
            if res["metrics"]["roc_auc"] > best_score:
                best_score = res["metrics"]["roc_auc"]
                best_candidate = res

    # Serialize best winning model
    if best_candidate is not None:
        logger.info(
            "Selected best model: '%s' with test ROC-AUC: %.4f and F1: %.4f",
            best_candidate["name"],
            best_candidate["metrics"]["roc_auc"],
            best_candidate["metrics"]["f1"],
        )
        joblib.dump(best_candidate["pipeline"], BEST_MODEL_PATH)
        logger.info("Saved best model pipeline to %s", BEST_MODEL_PATH)

        metadata = {
            "run_id": best_candidate["run_id"],
            "model_name": best_candidate["name"],
            "version": DEFAULT_MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": best_candidate["metrics"],
            "hyperparameters": {k.replace("classifier__", ""): str(v) for k, v in best_candidate["params"].items()},
            "leaderboard": [
                {
                    "model_name": r["name"],
                    "roc_auc": round(r["metrics"]["roc_auc"], 4),
                    "f1": round(r["metrics"]["f1"], 4),
                    "precision": round(r["metrics"]["precision"], 4),
                    "recall": round(r["metrics"]["recall"], 4),
                    "cv_roc_auc": round(r["metrics"].get("cv_roc_auc", 0.0), 4),
                }
                for r in results
            ],
        }
        with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved model metadata and leaderboard to %s", MODEL_METADATA_PATH)

    return {
        "best_model_name": best_candidate["name"] if best_candidate else None,
        "best_roc_auc": best_score,
        "runs": [
            {
                "run_id": r["run_id"],
                "name": r["name"],
                "metrics": r["metrics"],
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate customer churn models.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Path to source CSV dataset",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter optimization for rapid verification",
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=6,
        help="Number of iterations for hyperparameter search",
    )

    args = parser.parse_args()
    summary = run_training_suite(
        csv_path=Path(args.csv_path),
        tune_hyperparameters=not args.no_tune,
        tuning_iterations=args.tuning_iterations,
    )
    logger.info("Training suite completed successfully. Best model: %s", summary["best_model_name"])
