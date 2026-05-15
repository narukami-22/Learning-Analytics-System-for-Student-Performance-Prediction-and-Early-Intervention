"""
clustering_kmeans.py — K-Means Clustering
==========================================
Fixes from original:
  - Added Elbow Method (WCSS) to justify k=3
  - Added Silhouette Score and Davies-Bouldin evaluation
  - Added PCA 2D projection for visualization
  - Added cluster profiling (what each cluster means)
  - Added cluster labeling (At-Risk / Average / High Performer)
  - Saves cluster assignments back to data
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


def perform_clustering(data):
    # ── FEATURE SELECTION FOR CLUSTERING ─────────────────────────────────────
    cluster_features = [
        "G1", "G2", "failures", "absences", "studytime",
        "Dalc", "Walc", "goout", "health", "Medu", "Fedu",
        "higher", "internet", "freetime", "age"
    ]
    # Keep only columns present in data
    cluster_features = [f for f in cluster_features if f in data.columns]

    X = data[cluster_features].copy().fillna(0)

    # ── NORMALISATION ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── ELBOW METHOD — find optimal k ─────────────────────────────────────────
    # (original had no elbow method — k=3 was unjustified)
    wcss = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++",
                    n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    # Save WCSS for visualization
    np.save("wcss.npy", np.array(wcss))

    print("[Clustering] Elbow WCSS values (k=2 to 10):")
    for k, w in zip(k_range, wcss):
        print(f"  k={k}  WCSS={w:.2f}")

    # ── K-MEANS WITH k=3 ──────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=3, init="k-means++",
                    n_init=10, max_iter=300, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # ── EVALUATION METRICS ────────────────────────────────────────────────────
    # (original had none)
    sil  = silhouette_score(X_scaled, cluster_labels)
    db   = davies_bouldin_score(X_scaled, cluster_labels)
    print(f"\n[Clustering] k=3 Results:")
    print(f"  Silhouette Score   : {sil:.4f}  (higher better, max=1)")
    print(f"  Davies-Bouldin     : {db:.4f}   (lower better)")
    print(f"  WCSS (Inertia)     : {kmeans.inertia_:.2f}")
    sizes = dict(zip(*np.unique(cluster_labels, return_counts=True)))
    print(f"  Cluster sizes      : {sizes}")

    # ── PCA 2D PROJECTION ─────────────────────────────────────────────────────
    # (original had no PCA — needed for scatter plot)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    np.save("X_pca.npy", X_pca)
    print(f"  PCA 2D variance    : {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # ── CLUSTER LABELLING ─────────────────────────────────────────────────────
    # (original had no labels — clusters were just 0/1/2 with no meaning)
    data = data.copy()
    data["Cluster"] = cluster_labels

    cluster_g3 = data.groupby("Cluster")["G3"].mean().sort_values()
    sorted_clusters = cluster_g3.index.tolist()
    label_map = {
        sorted_clusters[0]: "At-Risk",
        sorted_clusters[1]: "Average",
        sorted_clusters[2]: "High Performer"
    }
    data["cluster_label"] = data["Cluster"].map(label_map)

    print(f"\n[Clustering] Cluster → Label mapping: {label_map}")
    print("[Clustering] Cluster Profile (mean values):")
    profile_cols = [c for c in ["G1","G2","G3","absences","failures",
                                 "studytime","goout","Dalc"] if c in data.columns]
    print(data.groupby("cluster_label")[profile_cols].mean().round(2).to_string())

    # Save cluster assignments for intervention module
    data[["student_id", "Cluster", "cluster_label"]].to_csv(
        "cluster_assignments.csv", index=False)
    print("\n[Clustering] Saved → cluster_assignments.csv\n")

    # Save full data with PCA components attached
    np.save("cluster_labels.npy", cluster_labels)

    return data
