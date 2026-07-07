"""
Customer & Transaction Management.

Manages the canonical raw data files (data/customers.csv and
data/transactions.csv) as the single source of truth. Provides
functions to add a new customer or a new transaction, with
automatic ID generation, duplicate/missing-value handling, and
recalculation of purchase aggregates (total purchases, total
spending, average order value, last purchase date).

Archetype assignment itself lives in archetype.py and is applied
on top of the engineered feature table built from these raw files,
so any add_customer/add_transaction call followed by a pipeline
rebuild automatically refreshes every customer's archetype.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = "data"
RAW_CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
RAW_TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")

CUSTOMER_COLUMNS = ["customer_id", "name", "age", "gender", "income", "city",
                     "signup_date", "support_tickets", "engagement_score",
                     "email_opt_in", "satisfaction_rating"]
TRANSACTION_COLUMNS = ["transaction_id", "customer_id", "transaction_date", "category",
                        "product_id", "quantity", "amount", "payment_method", "channel"]


def ensure_raw_files(default_customers: pd.DataFrame = None,
                      default_transactions: pd.DataFrame = None) -> None:
    """Create the canonical raw CSV files if they don't exist yet,
    seeding them with the provided (e.g. synthetic) data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_CUSTOMERS_PATH):
        (default_customers if default_customers is not None
         else pd.DataFrame(columns=CUSTOMER_COLUMNS)).to_csv(RAW_CUSTOMERS_PATH, index=False)
        logger.info("Created new %s", RAW_CUSTOMERS_PATH)
    if not os.path.exists(RAW_TRANSACTIONS_PATH):
        (default_transactions if default_transactions is not None
         else pd.DataFrame(columns=TRANSACTION_COLUMNS)).to_csv(RAW_TRANSACTIONS_PATH, index=False)
        logger.info("Created new %s", RAW_TRANSACTIONS_PATH)


def load_raw_customers() -> pd.DataFrame:
    df = pd.read_csv(RAW_CUSTOMERS_PATH, parse_dates=["signup_date"])
    df = _clean_customers_basic(df)
    return df


def load_raw_transactions() -> pd.DataFrame:
    df = pd.read_csv(RAW_TRANSACTIONS_PATH, parse_dates=["transaction_date"])
    df = _clean_transactions_basic(df)
    return df


def _clean_customers_basic(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["customer_id"])
    df["income"] = df["income"].fillna(df["income"].median() if len(df) else 0)
    df["age"] = df["age"].fillna(df["age"].median() if len(df) else 0)
    for col, default in [("support_tickets", 0), ("engagement_score", 50.0),
                          ("email_opt_in", 1), ("satisfaction_rating", 3.5)]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    if before != len(df):
        logger.info("Removed %d duplicate customer rows", before - len(df))
    return df


def _clean_transactions_basic(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["customer_id", "transaction_date", "amount"])
    df = df.drop_duplicates(subset=["transaction_id"])
    df = df[(df["amount"] > 0) & (df["quantity"] > 0)]
    if before != len(df):
        logger.info("Removed %d invalid/duplicate transaction rows", before - len(df))
    return df


def generate_customer_id(existing_ids: pd.Series) -> str:
    if len(existing_ids) == 0:
        return "CUST00001"
    nums = existing_ids.str.extract(r"CUST(\d+)").astype(float)
    next_num = int(nums.max().iloc[0]) + 1 if not nums.empty and not nums.isna().all().iloc[0] else 1
    return f"CUST{next_num:05d}"


def generate_transaction_id(existing_ids: pd.Series) -> str:
    if len(existing_ids) == 0:
        return "TXN0000001"
    nums = existing_ids.str.extract(r"TXN(\d+)").astype(float)
    next_num = int(nums.max().iloc[0]) + 1 if not nums.empty and not nums.isna().all().iloc[0] else 1
    return f"TXN{next_num:07d}"


def add_customer(name: str, age: int, gender: str, income: float, city: str,
                  email_opt_in: bool = True, signup_date: Optional[datetime] = None) -> str:
    """Append a new customer to customers.csv with auto-generated ID.
    Returns the new customer_id. Archetype is implicitly 'new' since
    the customer has zero transactions and a fresh signup date — the
    rule engine will assign that label on the next pipeline rebuild."""
    ensure_raw_files()
    customers = load_raw_customers()

    new_id = generate_customer_id(customers["customer_id"])
    new_row = {
        "customer_id": new_id,
        "name": name,
        "age": age,
        "gender": gender,
        "income": income,
        "city": city,
        "signup_date": signup_date or datetime.now(),
        "support_tickets": 0,
        "engagement_score": 50.0,
        "email_opt_in": 1 if email_opt_in else 0,
        "satisfaction_rating": 3.5,
    }
    customers = pd.concat([customers, pd.DataFrame([new_row])], ignore_index=True)
    customers.to_csv(RAW_CUSTOMERS_PATH, index=False)
    logger.info("Added new customer %s (%s)", new_id, name)
    return new_id


def add_transaction(customer_id: str, category: str, product_id: str, quantity: int,
                     amount: float, payment_method: str, channel: str,
                     transaction_date: Optional[datetime] = None) -> str:
    """Append a new transaction to transactions.csv with an
    auto-generated ID. Returns the new transaction_id. Caller is
    responsible for triggering a pipeline rebuild afterwards so the
    customer's archetype, KPIs, segmentation and churn model refresh."""
    ensure_raw_files()
    transactions = load_raw_transactions()

    new_id = generate_transaction_id(transactions["transaction_id"])
    new_row = {
        "transaction_id": new_id,
        "customer_id": customer_id,
        "transaction_date": transaction_date or datetime.now(),
        "category": category,
        "product_id": product_id,
        "quantity": quantity,
        "amount": amount,
        "payment_method": payment_method,
        "channel": channel,
    }
    transactions = pd.concat([transactions, pd.DataFrame([new_row])], ignore_index=True)
    transactions.to_csv(RAW_TRANSACTIONS_PATH, index=False)
    logger.info("Added new transaction %s for customer %s (₹%.2f)", new_id, customer_id, amount)
    return new_id


def customer_purchase_stats(customer_id: str, transactions: pd.DataFrame) -> dict:
    """Total purchases, total spending, average order value and last
    purchase date for a single customer — used to show a live summary
    right after a transaction is recorded."""
    cust_tx = transactions[transactions["customer_id"] == customer_id]
    if cust_tx.empty:
        return {"total_purchases": 0, "total_spending": 0.0,
                "avg_order_value": 0.0, "last_purchase_date": None}
    return {
        "total_purchases": int(len(cust_tx)),
        "total_spending": float(cust_tx["amount"].sum()),
        "avg_order_value": float(cust_tx["amount"].mean()),
        "last_purchase_date": cust_tx["transaction_date"].max(),
    }
