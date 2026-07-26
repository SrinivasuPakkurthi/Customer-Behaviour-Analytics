"""
Automatic Customer Archetype Management.

Replaces manual archetype assignment with rule-based classification
derived from each customer's signup recency and purchase history
(recency, frequency, monetary — already computed by preprocessing.py).

Archetypes (priority order, top rule wins):
    at_risk     -> has purchased before, but not in the last 90 days
    high_value  -> lifetime spending > HIGH_VALUE_THRESHOLD
    loyal       -> > LOYAL_MIN_PURCHASES purchases AND spending > LOYAL_MIN_SPEND
                    AND purchased within RECENT_WINDOW_DAYS
    regular     -> 1-10 purchases AND spending < LOYAL_MIN_SPEND
    new         -> signed up within NEW_CUSTOMER_WINDOW_DAYS with no/very few purchases
    (fallback)  -> regular
"""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds (same currency unit as the `amount` column in transactions.csv)
NEW_CUSTOMER_WINDOW_DAYS = 30
NEW_CUSTOMER_MAX_PURCHASES = 2          # "no or very few transactions"
RECENT_WINDOW_DAYS = 90                 # used by both `loyal` and `at_risk`
REGULAR_MIN_PURCHASES = 1
REGULAR_MAX_PURCHASES = 10
REGULAR_MAX_SPEND = 50_000
LOYAL_MIN_PURCHASES = 10
LOYAL_MIN_SPEND = 50_000
HIGH_VALUE_THRESHOLD = 150_000

VALID_ARCHETYPES = ["new", "regular", "loyal", "at_risk", "high_value"]


def classify_archetype(tenure_days: float, recency_days: float,
                        frequency: float, monetary: float) -> str:
    """Apply the archetype rules to a single customer's stats and
    return one of VALID_ARCHETYPES."""
    tenure_days = 0 if pd.isna(tenure_days) else tenure_days
    recency_days = 9999 if pd.isna(recency_days) else recency_days
    frequency = 0 if pd.isna(frequency) else frequency
    monetary = 0 if pd.isna(monetary) else monetary

    has_purchased = frequency > 0

    # at_risk: has a purchase history but gone quiet for 90+ days
    if has_purchased and recency_days > RECENT_WINDOW_DAYS:
        return "at_risk"

    # high_value: lifetime spend crosses the top threshold, regardless of recency
    if monetary > HIGH_VALUE_THRESHOLD:
        return "high_value"

    # loyal: frequent, high spend, and recently active
    if (frequency > LOYAL_MIN_PURCHASES and monetary > LOYAL_MIN_SPEND
            and recency_days <= RECENT_WINDOW_DAYS):
        return "loyal"

    # regular: some purchase history but modest spend
    if REGULAR_MIN_PURCHASES <= frequency <= REGULAR_MAX_PURCHASES and monetary < REGULAR_MAX_SPEND:
        return "regular"

    # new: recently signed up with little/no purchase history
    if tenure_days <= NEW_CUSTOMER_WINDOW_DAYS and frequency <= NEW_CUSTOMER_MAX_PURCHASES:
        return "new"

    return "regular"


def assign_archetypes(features: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Recalculate the `archetype` column for every customer in the
    engineered feature table (must already contain tenure_days,
    recency_days, frequency, monetary).
    """
    df = features.copy()
    df["archetype"] = df.apply(
        lambda r: classify_archetype(r.get("tenure_days"), r.get("recency_days"),
                                       r.get("frequency"), r.get("monetary")),
        axis=1,
    )
    logger.info("Archetype distribution: %s", df["archetype"].value_counts().to_dict())
    return df


def archetype_counts(features: pd.DataFrame) -> Dict[str, int]:
    counts = features["archetype"].value_counts().to_dict()
    return {a: int(counts.get(a, 0)) for a in VALID_ARCHETYPES}
