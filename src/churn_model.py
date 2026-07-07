"""
Churn prediction: trains Logistic Regression, Decision Tree and
Random Forest, compares them, and selects/saves the best model.
"""

import logging
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

logger = logging.getLogger(__name__)

MODEL_FEATURES = ["age", "income", "recency_days", "frequency", "monetary",
                   "avg_order_value", "tenure_days", "purchase_frequency_rate",
                   "RFM_score", "CLV", "support_tickets", "engagement_score",
                   "satisfaction_rating"]


def prepare_model_data(
    features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, pd.Series]:
    """Split customer features into scaled train/test sets for churn modelling."""
    df = features.dropna(subset=MODEL_FEATURES + ["churned"]).copy()
    logger.info("Preparing model data: %d customers, %d features", len(df), len(MODEL_FEATURES))
    X = df[MODEL_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, df.loc[X_test.index, "customer_id"]


def train_and_compare(features: pd.DataFrame) -> Dict[str, Any]:
    """Train Logistic Regression, Decision Tree and Random Forest churn models,
    compare them on standard classification metrics, and return the best one
    along with comparison data for display."""
    X_train, X_test, y_train, y_test, scaler, test_ids = prepare_model_data(features)
    logger.info("Training churn models on %d training rows", len(y_train))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }

    results = {}
    fitted_models = {}
    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "fpr_tpr": roc_curve(y_test, probs)[:2],
        }
        fitted_models[name] = model
        logger.info("%s -> accuracy=%.4f roc_auc=%.4f", name,
                     results[name]["accuracy"], results[name]["roc_auc"])

    comparison_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k not in ("confusion_matrix", "fpr_tpr")}
        for name, r in results.items()
    }).T.round(4)

    best_name = comparison_df["roc_auc"].idxmax()
    best_model = fitted_models[best_name]

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=MODEL_FEATURES)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_[0]), index=MODEL_FEATURES)
    else:
        importances = pd.Series(0, index=MODEL_FEATURES)
    importances = importances.sort_values(ascending=False)

    return {
        "results": results,
        "comparison_df": comparison_df,
        "best_name": best_name,
        "best_model": best_model,
        "scaler": scaler,
        "feature_importances": importances,
        "y_test": y_test,
        "test_ids": test_ids,
    }


def save_model(model: Any, scaler: StandardScaler, path_model: str = "models/churn_model.joblib",
               path_scaler: str = "models/scaler.joblib") -> None:
    """Persist the trained model and its feature scaler to disk."""
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
    logger.info("Saved model to %s and scaler to %s", path_model, path_scaler)


def load_model(path_model: str = "models/churn_model.joblib",
                path_scaler: str = "models/scaler.joblib") -> Tuple[Any, StandardScaler]:
    """Load a previously saved model and scaler from disk."""
    return joblib.load(path_model), joblib.load(path_scaler)


def predict_churn_probability(model: Any, scaler: StandardScaler, customer_row: pd.DataFrame) -> float:
    """Predict churn probability for a single customer row."""
    X = customer_row[MODEL_FEATURES]
    X_scaled = scaler.transform(X)
    return float(model.predict_proba(X_scaled)[:, 1][0])


def risk_level(prob: float) -> str:
    """Map a churn probability to a human-readable risk level."""
    if prob >= 0.66:
        return "High Risk"
    if prob >= 0.33:
        return "Medium Risk"
    return "Low Risk"
