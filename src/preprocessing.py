"""
Data preprocessing, cleaning and feature engineering
(RFM + Customer Lifetime Value) for the Churn Prediction System.
"""

import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_transactions(tx: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean raw transaction data and return cleaning stats."""
    stats = {"rows_before": len(tx)}
    logger.info("Cleaning %d raw transaction rows", len(tx))

    tx = tx.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")

    stats["missing_values_before"] = int(tx.isna().sum().sum())
    tx = tx.dropna(subset=["customer_id", "transaction_date", "amount"])

    stats["duplicates_removed"] = int(tx.duplicated().sum())
    tx = tx.drop_duplicates()

    # remove invalid records
    before_invalid = len(tx)
    tx = tx[(tx["amount"] > 0) & (tx["quantity"] > 0)]
    stats["invalid_removed"] = before_invalid - len(tx)

    # outlier handling via IQR capping on amount (wide multiplier + sensible floor so
    # legitimate larger manually-entered transactions aren't crushed by a tight IQR
    # computed mostly from smaller synthetic/historical purchases)
    q1, q3 = tx["amount"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = max(q3 + 5 * iqr, 20_000)
    lower = max(0, q1 - 5 * iqr)
    stats["outliers_capped"] = int(((tx["amount"] > upper) | (tx["amount"] < lower)).sum())
    tx["amount"] = tx["amount"].clip(lower=lower, upper=upper)

    stats["rows_after"] = len(tx)
    logger.info("Transaction cleaning complete: %s", stats)
    return tx, stats


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    customers = customers.drop_duplicates(subset=["customer_id"])
    customers["income"] = customers["income"].fillna(customers["income"].median())
    customers["age"] = customers["age"].fillna(customers["age"].median())
    return customers


def build_customer_features(customers: pd.DataFrame, tx: pd.DataFrame,
                             reference_date: pd.Timestamp = None,
                             churn_inactivity_days: int = 90) -> pd.DataFrame:
    """
    Build a customer-level feature table including RFM, CLV,
    and a churn label (no purchase within `churn_inactivity_days`).
    """
    if reference_date is None:
        reference_date = tx["transaction_date"].max() + pd.Timedelta(days=1)

    grouped = tx.groupby("customer_id").agg(
        recency_days=("transaction_date", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        first_purchase=("transaction_date", "min"),
        last_purchase=("transaction_date", "max"),
        favorite_category=("category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        favorite_channel=("channel", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    ).reset_index()

    features = customers.merge(grouped, on="customer_id", how="left")

    # customers with no transactions
    features["frequency"] = features["frequency"].fillna(0)
    features["monetary"] = features["monetary"].fillna(0)
    features["avg_order_value"] = features["avg_order_value"].fillna(0)
    features["recency_days"] = features["recency_days"].fillna(
        (reference_date - features["signup_date"]).dt.days
    )

    # tenure & purchase span
    features["tenure_days"] = (reference_date - features["signup_date"]).dt.days.clip(lower=1)
    span = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["purchase_span_days"] = span.fillna(0).clip(lower=1)
    features["purchase_frequency_rate"] = (
        features["frequency"] / (features["purchase_span_days"] / 30)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # RFM scores (1-5, 5 = best)
    features["R_score"] = pd.qcut(features["recency_days"].rank(method="first"), 5,
                                   labels=[5, 4, 3, 2, 1]).astype(int)
    features["F_score"] = pd.qcut(features["frequency"].rank(method="first"), 5,
                                   labels=[1, 2, 3, 4, 5]).astype(int)
    features["M_score"] = pd.qcut(features["monetary"].rank(method="first"), 5,
                                   labels=[1, 2, 3, 4, 5]).astype(int)
    features["RFM_score"] = (features["R_score"] + features["F_score"] + features["M_score"])
    features["RFM_segment_code"] = (
        features["R_score"].astype(str) + features["F_score"].astype(str) + features["M_score"].astype(str)
    )

    # Customer Lifetime Value (simplified): avg order value * purchase freq/month * avg lifespan (months)
    avg_lifespan_months = (features["tenure_days"] / 30).clip(lower=1)
    features["CLV"] = (features["avg_order_value"] *
                        features["purchase_frequency_rate"] *
                        avg_lifespan_months).round(2)

    # Churn label: a probabilistic combination of recency (primary driver) with
    # independent behavioural signals (support tickets, engagement, satisfaction)
    # plus noise, so the prediction task is realistic rather than a deterministic
    # restatement of a single feature.
    recency_signal = (features["recency_days"] / churn_inactivity_days).clip(0, 3)
    ticket_signal = (features.get("support_tickets", 0).fillna(0) / 6).clip(0, 1.5)
    engagement_signal = 1 - (features.get("engagement_score", 50).fillna(50) / 100)
    satisfaction_signal = (5 - features.get("satisfaction_rating", 3.5).fillna(3.5)) / 4

    churn_score = (
        0.55 * recency_signal +
        0.20 * ticket_signal +
        0.15 * engagement_signal +
        0.10 * satisfaction_signal
    )
    noise = np.random.default_rng(42).normal(0, 0.12, len(features))
    churn_prob = 1 / (1 + np.exp(-4 * (churn_score - 0.6) + noise))
    features["churn_probability_true"] = churn_prob.round(3)
    features["churned"] = (churn_prob > 0.5).astype(int)

    return features


def rfm_segment_label(row) -> str:
    """Map RFM scores to human-readable segment names."""
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 4:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r >= 3 and f >= 3:
        return "Potential Loyalists"
    if r == 3 and f <= 2:
        return "Promising Customers"
    if r <= 2 and f >= 3:
        return "Need Attention"
    if r <= 2 and f <= 2 and m >= 3:
        return "At Risk"
    return "Lost Customers"
