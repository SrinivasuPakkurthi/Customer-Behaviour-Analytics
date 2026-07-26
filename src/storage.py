"""
Persistence layer: saves and loads processed customer/transaction
data and trained models to/from disk so application state survives
a restart.
"""

import json
import logging
import os
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODELS_DIR = "models"
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers_processed.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions_processed.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "customer_features.csv")
META_PATH = os.path.join(DATA_DIR, "pipeline_meta.json")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def save_pipeline_state(customers: pd.DataFrame, tx: pd.DataFrame,
                          features: pd.DataFrame, clean_stats: dict) -> None:
    """Persist the cleaned customers, transactions and engineered
    feature table to disk so they can be reloaded without re-running
    the full pipeline."""
    ensure_dirs()
    customers.to_csv(CUSTOMERS_PATH, index=False)
    tx.to_csv(TRANSACTIONS_PATH, index=False)
    features.to_csv(FEATURES_PATH, index=False)
    with open(META_PATH, "w") as f:
        json.dump(clean_stats, f, indent=2)
    logger.info("Pipeline state saved to %s", DATA_DIR)


def load_pipeline_state() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]]:
    """Load a previously persisted pipeline state, if one exists."""
    if not all(os.path.exists(p) for p in
                (CUSTOMERS_PATH, TRANSACTIONS_PATH, FEATURES_PATH, META_PATH)):
        return None
    try:
        customers = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date"])
        tx = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["transaction_date"])
        features = pd.read_csv(
            FEATURES_PATH, parse_dates=["signup_date", "first_purchase", "last_purchase"]
        )
        with open(META_PATH) as f:
            clean_stats = json.load(f)
        logger.info("Pipeline state loaded from %s", DATA_DIR)
        return customers, tx, features, clean_stats
    except Exception as exc:
        logger.warning("Failed to load persisted pipeline state: %s", exc)
        return None


def clear_pipeline_state() -> None:
    for path in (CUSTOMERS_PATH, TRANSACTIONS_PATH, FEATURES_PATH, META_PATH):
        if os.path.exists(path):
            os.remove(path)
    logger.info("Cleared persisted pipeline state")


def saved_model_exists(path_model: str = os.path.join(MODELS_DIR, "churn_model.joblib")) -> bool:
    return os.path.exists(path_model)
