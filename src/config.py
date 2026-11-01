"""Central configuration and constants for churn-mlops-platform.

Loads environment variables, centralizes paths, column definitions,
hyperparameters, and configures standardized structured logging.
"""

from pathlib import Path
import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(dotenv_path=BASE_DIR / ".env")

# MLflow environment settings for local filestore tracking
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Supabase Credentials
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://dummy.supabase.co")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "dummy-anon-key")

# MLflow Tracking
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"file:{(BASE_DIR / 'mlruns').as_posix()}")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")

# Local Storage Paths
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "telco_churn.csv"
MODELS_DIR: Path = BASE_DIR / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
SHAP_SUMMARY_PATH: Path = MODELS_DIR / "shap_summary.png"
SHAP_BAR_PATH: Path = MODELS_DIR / "shap_bar.png"

# Ensure essential local directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Column Definitions
ID_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"

NUMERICAL_COLS: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Service subscription indicator columns used for feature engineering
SERVICE_COLS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Risk classification thresholds
RISK_THRESHOLD_LOW: float = 0.35
RISK_THRESHOLD_HIGH: float = 0.70

# Default model version
DEFAULT_MODEL_VERSION: str = "v1.0.0"

# Application runtime settings
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a standardized logger instance.

    Args:
        name: Name of the logger, typically __name__.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
