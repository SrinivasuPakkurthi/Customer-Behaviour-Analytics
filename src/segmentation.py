"""
Customer segmentation via K-Means clustering, including
Elbow method, silhouette score, and PCA visualisation prep.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLUSTER_FEATURES = ["recency_days", "frequency", "monetary", "CLV"]


def elbow_method(features: pd.DataFrame, k_range=range(2, 11)):
    X = StandardScaler().fit_transform(features[CLUSTER_FEATURES])
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
    return list(k_range), inertias, sil_scores


def run_kmeans(features: pd.DataFrame, n_clusters: int = 4):
    X = StandardScaler().fit_transform(features[CLUSTER_FEATURES])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = features.copy()
    out["cluster"] = labels
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    return out, sil, km


def label_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters by monetary value and assign business-friendly names."""
    profile = df.groupby("cluster")[CLUSTER_FEATURES].mean()
    profile["score"] = (
        profile["monetary"].rank() + profile["frequency"].rank() - profile["recency_days"].rank()
    )
    ordered = profile.sort_values("score", ascending=False).index.tolist()

    names = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers",
              "Promising Customers", "Need Attention", "At Risk", "Lost Customers"]
    mapping = {cluster_id: names[i] if i < len(names) else f"Segment {i+1}"
               for i, cluster_id in enumerate(ordered)}

    out = df.copy()
    out["segment"] = out["cluster"].map(mapping)
    return out


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_spending=("monetary", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_recency=("recency_days", "mean"),
        revenue_contribution=("monetary", "sum"),
        churn_rate=("churned", "mean"),
    ).reset_index()
    summary["revenue_share_%"] = (summary["revenue_contribution"] /
                                   summary["revenue_contribution"].sum() * 100).round(2)
    summary["churn_rate_%"] = (summary["churn_rate"] * 100).round(2)
    return summary.sort_values("revenue_contribution", ascending=False)
