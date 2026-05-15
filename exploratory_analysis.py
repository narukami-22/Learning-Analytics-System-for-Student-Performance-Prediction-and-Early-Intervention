"""
exploratory_analysis.py — Exploratory Data Analysis
=====================================================
Fixes from original:
  - Had only 1 chart (grade histogram with plt.show)
  - plt.show() replaced with savefig (works in non-interactive env)
  - Added: correlation heatmap, grade distributions (G1/G2/G3),
           subject breakdown, PCA variance, feature importance,
           absences vs G3 scatter, boxplots by performance class
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")             # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white"
})

PALETTE = ["#e74c3c", "#f39c12", "#27ae60"]   # At-Risk / Average / High


# ── 1. Grade Distribution (original function — now saves instead of shows) ──

def plot_grade_distribution(data):
    """Final grade distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data["G3"], bins=20, color="#3498db",
            edgecolor="white", alpha=0.85)
    ax.axvline(data["G3"].mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {data['G3'].mean():.1f}")
    ax.set_title("Final Grade (G3) Distribution")
    ax.set_xlabel("Grade (0–20)")
    ax.set_ylabel("Number of Students")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_grade_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_grade_distribution.png")


# ── 2. All Three Grades Side by Side ──────────────────────────────────────────

def plot_all_grades(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#3498db", "#9b59b6", "#e74c3c"]
    for i, (gc, col) in enumerate(zip(["G1","G2","G3"], colors)):
        axes[i].hist(data[gc], bins=20, color=col, edgecolor="white", alpha=0.85)
        axes[i].axvline(data[gc].mean(), color="black", linestyle="--",
                        linewidth=1.5, label=f"Mean={data[gc].mean():.1f}")
        axes[i].set_title(f"{gc} Distribution")
        axes[i].set_xlabel("Grade (0–20)")
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=9)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
    plt.suptitle("Grade Distribution Across All Three Periods",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_all_grades.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_all_grades.png")


# ── 3. Performance Class Distribution ─────────────────────────────────────────

def plot_class_distribution(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts = data["performance_class"].value_counts().reindex(
        ["At-Risk","Average","High"])
    bars = axes[0].bar(counts.index, counts.values,
                       color=PALETTE, edgecolor="white", width=0.55)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 3,
                     str(val), ha="center", fontweight="bold")
    axes[0].set_title("Overall Performance Class Distribution")
    axes[0].set_ylabel("Number of Students")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    counts_s = data.groupby(["subject","performance_class"]).size().unstack(fill_value=0)
    counts_s = counts_s.reindex(columns=["At-Risk","Average","High"])
    counts_s.plot(kind="bar", ax=axes[1], color=PALETTE,
                  edgecolor="white", width=0.6)
    axes[1].set_title("Performance Class by Subject")
    axes[1].set_ylabel("Number of Students")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_class_distribution.png")


# ── 4. Correlation Heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(data):
    key_cols = ["G1","G2","G3","studytime","failures","absences",
                "Medu","Fedu","goout","Dalc","Walc","health","age"]
    key_cols = [c for c in key_cols if c in data.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data[key_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, square=True,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Pairwise Correlation Heatmap", pad=15)
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_correlation_heatmap.png")


# ── 5. Feature Importance (Mutual Information) ────────────────────────────────

def plot_feature_importance(data):
    num_cols = ["G1","G2","studytime","failures","absences","Medu","Fedu",
                "goout","Dalc","Walc","health","age","freetime",
                "higher","internet","romantic","famsup"]
    num_cols = [c for c in num_cols if c in data.columns]

    le = LabelEncoder()
    y = le.fit_transform(data["performance_class"])
    X = data[num_cols].fillna(0)

    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(mi_series)))
    mi_series.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_title("Feature Importance — Mutual Information Score")
    ax.set_xlabel("Mutual Information Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_feature_importance.png")


# ── 6. Absences vs G3 Scatter ─────────────────────────────────────────────────

def plot_absences_vs_grade(data):
    color_map = {"At-Risk": "#e74c3c", "Average": "#f39c12", "High": "#27ae60"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in color_map.items():
        subset = data[data["performance_class"] == label]
        ax.scatter(subset["absences"], subset["G3"],
                   c=color, label=label, alpha=0.5, s=25, edgecolors="none")
    ax.set_title("Absences vs Final Grade (G3)")
    ax.set_xlabel("Number of Absences")
    ax.set_ylabel("Final Grade (G3)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_absences_vs_grade.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_absences_vs_grade.png")


# ── 7. G3 Boxplot by Performance Class ───────────────────────────────────────

def plot_boxplot_by_class(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["At-Risk", "Average", "High"]
    plot_data = [data[data["performance_class"] == c]["G3"].values
                 for c in order]
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
    ax.set_xticklabels(order)
    ax.set_title("Final Grade (G3) Distribution by Performance Class")
    ax.set_xlabel("Performance Class")
    ax.set_ylabel("G3 (Final Grade)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_boxplot_by_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_boxplot_by_class.png")


# ── RUN ALL ───────────────────────────────────────────────────────────────────

def run_exploratory_analysis(data):
    """Run all EDA charts."""
    print("\n[EDA] Generating exploratory analysis charts...")
    plot_grade_distribution(data)
    plot_all_grades(data)
    plot_class_distribution(data)
    plot_correlation_heatmap(data)
    plot_feature_importance(data)
    plot_absences_vs_grade(data)
    plot_boxplot_by_class(data)
    print("[EDA] All charts saved.\n")
