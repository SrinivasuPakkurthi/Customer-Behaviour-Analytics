"""
Synthetic customer transaction data generator.
Creates a realistic e-commerce style dataset for the
Customer Behaviour Analysis & Churn Prediction System.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)


def generate_customers(n_customers: int = 1000) -> pd.DataFrame:
    """Generate base customer profile data."""
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer",
                   "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
                   "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
                   "Charles", "Karen", "Arjun", "Priya", "Wei", "Mei", "Hiro",
                   "Sofia", "Liam", "Olivia", "Noah", "Emma"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
                  "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez",
                  "Thompson", "Patel", "Kim", "Nguyen", "Khan", "Silva"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
              "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
              "Mumbai", "Delhi", "Bangalore", "London", "Toronto", "Sydney"]
    genders = ["Male", "Female", "Other"]

    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    signup_start = datetime(2021, 1, 1)
    signup_end = datetime(2024, 12, 31)
    signup_days = (signup_end - signup_start).days

    data = {
        "customer_id": customer_ids,
        "name": [f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                  for _ in range(n_customers)],
        "age": np.random.randint(18, 70, n_customers),
        "gender": np.random.choice(genders, n_customers, p=[0.48, 0.48, 0.04]),
        "income": np.round(np.random.normal(55000, 20000, n_customers).clip(15000, 200000), 2),
        "city": np.random.choice(cities, n_customers),
        "signup_date": [signup_start + timedelta(days=int(np.random.randint(0, signup_days)))
                         for _ in range(n_customers)],
        # independent behavioural signals (not derived from transactions) so that
        # churn does not become a trivial function of recency alone
        "support_tickets": np.random.poisson(1.2, n_customers).clip(0, 12),
        "engagement_score": np.round(np.random.beta(2, 2, n_customers) * 100, 1),
        "email_opt_in": np.random.choice([1, 0], n_customers, p=[0.7, 0.3]),
        "satisfaction_rating": np.round(np.random.normal(3.7, 0.9, n_customers).clip(1, 5), 1),
    }
    return pd.DataFrame(data)


def generate_transactions(customers: pd.DataFrame, max_transactions: int = 60) -> pd.DataFrame:
    """Generate transaction-level data per customer."""
    categories = ["Electronics", "Fashion", "Grocery", "Home & Kitchen",
                  "Beauty", "Sports", "Books", "Toys", "Automotive", "Health"]
    payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking",
                        "Cash on Delivery", "Wallet"]
    channels = ["Web", "Mobile App", "In-Store"]
    products = {cat: [f"{cat[:3].upper()}-{i}" for i in range(1, 9)] for cat in categories}

    today = datetime.now()
    rows = []

    # assign a "behaviour archetype" per customer to create realistic churn signal
    archetypes = np.random.choice(
        ["champion", "loyal", "potential", "new", "at_risk", "lost"],
        size=len(customers), p=[0.10, 0.18, 0.17, 0.15, 0.20, 0.20]
    )

    for idx, (_, cust) in enumerate(customers.iterrows()):
        archetype = archetypes[idx]
        signup = cust["signup_date"]

        if archetype == "champion":
            n_tx = np.random.randint(30, max_transactions)
            recency_days = np.random.randint(0, 15)
            spend_mu = 180
        elif archetype == "loyal":
            n_tx = np.random.randint(20, 40)
            recency_days = np.random.randint(0, 30)
            spend_mu = 120
        elif archetype == "potential":
            n_tx = np.random.randint(8, 20)
            recency_days = np.random.randint(10, 60)
            spend_mu = 80
        elif archetype == "new":
            n_tx = np.random.randint(1, 6)
            recency_days = np.random.randint(0, 30)
            spend_mu = 70
        elif archetype == "at_risk":
            n_tx = np.random.randint(5, 15)
            recency_days = np.random.randint(60, 150)
            spend_mu = 90
        else:  # lost
            n_tx = np.random.randint(1, 8)
            recency_days = np.random.randint(150, 365)
            spend_mu = 60

        last_purchase = today - timedelta(days=int(recency_days))
        span_days = max((last_purchase - signup).days, 1)

        for _ in range(n_tx):
            offset = np.random.randint(0, span_days + 1)
            tx_date = signup + timedelta(days=int(offset))
            category = np.random.choice(categories)
            amount = max(5, np.random.normal(spend_mu, spend_mu * 0.4))
            rows.append({
                "transaction_id": f"TXN{len(rows)+1:07d}",
                "customer_id": cust["customer_id"],
                "transaction_date": tx_date,
                "category": category,
                "product_id": np.random.choice(products[category]),
                "quantity": np.random.randint(1, 5),
                "amount": round(amount, 2),
                "payment_method": np.random.choice(payment_methods),
                "channel": np.random.choice(channels, p=[0.5, 0.4, 0.1]),
            })

    tx_df = pd.DataFrame(rows)
    tx_df["amount"] = tx_df["amount"] * tx_df["quantity"] / tx_df["quantity"].mean()
    tx_df["amount"] = tx_df["amount"].round(2)
    return tx_df, archetypes


def generate_dataset(n_customers: int = 1000):
    customers = generate_customers(n_customers)
    transactions, _behaviour_archetypes = generate_transactions(customers)
    # NOTE: archetype is no longer set here. It is automatically computed
    # from purchase history by src/archetype.py (see assign_archetypes()).
    return customers, transactions


if __name__ == "__main__":
    customers, transactions = generate_dataset(1000)
    customers.to_csv("data/customers.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(customers)} customers and {len(transactions)} transactions")
