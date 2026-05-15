"""
visualization_dashboard.py — Results Visualization Dashboard
=============================================================
Fixes from original:
  - Had only 1 chart (cluster count with plt.show)
  - plt.show() replaced with savefig
  - Added: elbow curve, PCA scatter, cluster profile heatmap,
           classifier comparison, confusion matrix, metrics bar chart,
           G3 boxplot per cluster, intervention urgency chart
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white"
})

PALETTE   = ["#e74c3c", "#f39c12", "#27ae60"]
CLR_MAP   = {"At-Risk": "#e74c3c", "Average": "#f39c12",
             "High Performer": "#27ae60", "High": "#27ae60"}


# ── 1. Cluster Count (original function — now saves instead of shows) ─────────

def show_cluster_chart(data):
    """Bar chart of students per cluster."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = data["cluster_label"].value_counts() \
             if "cluster_label" in data.columns \
             else data["Cluster"].value_counts()
    colors = [CLR_MAP.get(str(c), "#95a5a6") for c in counts.index]
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Student Clusters — Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Students")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("viz_cluster_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_cluster_chart.png")


# ── 2. Elbow Curve ────────────────────────────────────────────────────────────

def plot_elbow_curve():
    if not __import__("os").path.exists("wcss.npy"):
        print("[VIZ] wcss.npy not found — skipping elbow curve")
        return
    wcss    = np.load("wcss.npy")
    k_range = range(2, 2 + len(wcss))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_range, wcss, "o-", color="#2c3e50", linewidth=2.5, markersize=8)
    ax.axvline(3, color="#e74c3c", linestyle="--",
               linewidth=1.5, label="Optimal k=3")
    ax.fill_between(k_range, wcss, alpha=0.08, color="#3498db")
    ax.set_title("Elbow Method — Optimal Number of Clusters (WCSS)")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Within-Cluster Sum of Squares")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("viz_elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_elbow_curve.png")


# ── 3. K-Means PCA Scatter ────────────────────────────────────────────────────

def plot_kmeans_scatter(data):
    if not __import__("os").path.exists("X_pca.npy"):
        print("[VIZ] X_pca.npy not found — skipping scatter plot")
        return
    X_pca = np.load("X_pca.npy")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Cluster labels scatter
    if "cluster_label" in data.columns:
        unique_labels = data["cluster_label"].unique()
        for label in unique_labels:
            mask = data["cluster_label"] == label
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=CLR_MAP.get(label, "#95a5a6"),
                            label=label, alpha=0.6, s=30, edgecolors="none")
    axes[0].set_title("K-Means Clusters (PCA 2D)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=1.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # True performance class scatter
    if "performance_class" in data.columns:
        for label, color in [("At-Risk","#e74c3c"),
                              ("Average","#f39c12"),
                              ("High","#27ae60")]:
            mask = data["performance_class"] == label
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=color, label=label, alpha=0.6,
                            s=30, edgecolors="none")
    axes[1].set_title("True Performance Labels (PCA 2D)")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(markerscale=1.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("K-Means Clusters vs True Labels",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("viz_kmeans_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_kmeans_scatter.png")


# ── 4. Cluster Profile Heatmap ────────────────────────────────────────────────

def plot_cluster_profile(data):
    profile_cols = [c for c in ["G1","G2","G3","absences","failures",
                                 "studytime","goout","Dalc","health"]
                    if c in data.columns]
    if "cluster_label" not in data.columns:
        print("[VIZ] cluster_label not found — skipping profile heatmap")
        return
    profile = data.groupby("cluster_label")[profile_cols].mean()
    order   = [c for c in ["At-Risk","Average","High Performer"]
               if c in profile.index]
    profile = profile.reindex(order)
    profile_norm = (profile - profile.min()) / (
                    profile.max() - profile.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(profile_norm.T, annot=profile.T.round(2), fmt=".2f",
                cmap="RdYlGn", linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Cluster Profile Heatmap (Normalised Feature Means)")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig("viz_cluster_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_cluster_profile.png")


# ── 5. Classifier Accuracy Bar ────────────────────────────────────────────────

def plot_classifier_comparison():
    import os
    if not os.path.exists("nn_metrics.csv"):
        print("[VIZ] nn_metrics.csv not found — skipping classifier chart")
        return
    df = pd.read_csv("nn_metrics.csv").sort_values("Accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df)))
    bars = ax.barh(df["Classifier"], df["Accuracy"],
                   color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, df["Accuracy"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontweight="bold", fontsize=10)
    ax.set_title("Classifier Accuracy Comparison")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("viz_classifier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_classifier_comparison.png")


# ── 6. All Metrics Bar Chart ──────────────────────────────────────────────────

def plot_all_metrics():
    import os
    if not os.path.exists("nn_metrics.csv"):
        return
    df = pd.read_csv("nn_metrics.csv")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    metric_colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (metric, color) in enumerate(zip(metric_cols, metric_colors)):
        ax.bar(x + i*width, df[metric], width, label=metric,
               color=color, edgecolor="white", alpha=0.85)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(df["Classifier"], rotation=20, ha="right")
    ax.set_title("Full Metrics Comparison Across Classifiers")
    ax.set_ylabel("Score (%)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("viz_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_all_metrics.png")


# ── 7. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix():
    import os
    if not os.path.exists("confusion_matrix.npy"):
        print("[VIZ] confusion_matrix.npy not found — skipping")
        return
    cm = np.load("confusion_matrix.npy")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["At-Risk","Average","High"],
                yticklabels=["At-Risk","Average","High"],
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — FDN Classifier")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("viz_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_confusion_matrix.png")


# ── 8. G3 Boxplot per Cluster ─────────────────────────────────────────────────

def plot_g3_by_cluster(data):
    if "cluster_label" not in data.columns:
        return
    order = [c for c in ["At-Risk","Average","High Performer"]
             if c in data["cluster_label"].unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = [data[data["cluster_label"] == c]["G3"].values for c in order]
    bp = ax.boxplot(plot_data, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
    ax.set_xticklabels(order)
    ax.set_title("Final Grade (G3) Distribution per Cluster")
    ax.set_xlabel("Cluster"); ax.set_ylabel("G3 (Final Grade)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("viz_g3_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_g3_boxplot.png")


# ── 9. Intervention Urgency ───────────────────────────────────────────────────

def plot_intervention_urgency(data):
    if "urgency" not in data.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts = data["urgency"].value_counts().reindex(
        ["High","Medium","Low"]).fillna(0)
    colors_u = ["#e74c3c","#f39c12","#27ae60"]
    bars = axes[0].bar(counts.index, counts.values,
                       color=colors_u, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1,
                     int(val), ha="center", fontweight="bold")
    axes[0].set_title("Intervention Urgency Distribution")
    axes[0].set_ylabel("Number of Students")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    itv = data["intervention"].str.split(" | ").explode().value_counts().head(8)
    itv.plot(kind="barh", ax=axes[1], color="#3498db", edgecolor="white")
    axes[1].set_title("Top 8 Interventions Recommended")
    axes[1].set_xlabel("Number of Students")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("viz_intervention_urgency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[VIZ] Saved → viz_intervention_urgency.png")


# ── RUN ALL ───────────────────────────────────────────────────────────────────

def run_dashboard(data):
    """Generate all dashboard charts."""
    print("\n[VIZ] Generating visualization dashboard...")
    show_cluster_chart(data)
    plot_elbow_curve()
    plot_kmeans_scatter(data)
    plot_cluster_profile(data)
    plot_classifier_comparison()
    plot_all_metrics()
    plot_confusion_matrix()
    plot_g3_by_cluster(data)
    plot_intervention_urgency(data)
    print("[VIZ] All charts saved.\n")
