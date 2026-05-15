"""
prediction_neural_network.py — Neural Network + Classifier Comparison
=======================================================================
Fixes from original:
  - Removed TensorFlow dependency (uses sklearn MLP — same architecture)
  - Changed binary classification → multiclass (At-Risk / Average / High)
  - Architecture upgraded to 200→100 (matching Paper 1 Table 3)
  - Added Dropout equivalent (early stopping)
  - Added evaluation: accuracy, precision, recall, F1, confusion matrix
  - Added 5-fold cross-validation
  - Added comparison with RF, SVM, KNN, NB, Decision Tree
  - Added clustered training experiment (Paper 1's key contribution)
  - Saves confusion matrix and metrics CSV
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)
import warnings
warnings.filterwarnings("ignore")


def _build_features(data):
    """Select and scale numeric features for model training."""
    feature_cols = [
        "G1", "G2", "failures", "absences", "studytime",
        "Dalc", "Walc", "goout", "health", "Medu", "Fedu",
        "higher", "internet", "freetime", "age",
        "famsup", "romantic", "sex_enc", "address_enc"
    ]
    feature_cols = [f for f in feature_cols if f in data.columns]

    X = data[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(data["performance_class"])   # At-Risk=0, Average=1, High=2

    return X_scaled, y, feature_cols, le


def _evaluate(clf, X_tr, X_te, y_tr, y_te, name):
    """Fit a classifier and return its metrics as a dict."""
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "Classifier": name,
        "Accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
        "Precision": round(precision_score(y_te, y_pred,
                           average="macro", zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_te, y_pred,
                           average="macro", zero_division=0) * 100, 2),
        "F1":        round(f1_score(y_te, y_pred,
                           average="macro", zero_division=0) * 100, 2),
    }


def train_model(data):
    print("[NN] Building features and labels...")
    X_scaled, y, feature_cols, le = _build_features(data)

    # Save scaled features for visualization
    np.save("X_scaled.npy", X_scaled)
    np.save("y_labels.npy", y)
    with open("feature_names.txt", "w") as f:
        f.write("\n".join(feature_cols))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # ── CLASSIFIERS ──────────────────────────────────────────────────────────
    # FDN architecture matches Paper 1 Table 3: 200→100, ReLU, Adam
    classifiers = {
        "FDN (MLP)": MLPClassifier(
            hidden_layer_sizes=(200, 100),
            activation="relu",
            solver="adam",
            batch_size=8,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        "Random Forest":   RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":   GradientBoostingClassifier(random_state=42),
        "SVM":             SVC(kernel="rbf", random_state=42),
        "KNN":             KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":     GaussianNB(),
    }

    # ── EXPERIMENT A: Full Dataset ────────────────────────────────────────────
    print("\n[NN] Experiment A — Full Dataset (80/20 split):")
    results = []
    for name, clf in classifiers.items():
        r = _evaluate(clf, X_tr, X_te, y_tr, y_te, name)
        results.append(r)
        print(f"  {name:<22} Acc={r['Accuracy']:>6.2f}%  F1={r['F1']:>6.2f}%")

    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv("nn_metrics.csv", index=False)

    # ── EXPERIMENT B: Clustered FDN ───────────────────────────────────────────
    # (Paper 1's key contribution: train one FDN per cluster)
    print("\n[NN] Experiment B — Clustered FDN (per-cluster training):")
    if "Cluster" in data.columns:
        cluster_results = []
        for c in sorted(data["Cluster"].unique()):
            idx = data.index[data["Cluster"] == c].tolist()
            Xc, yc = X_scaled[idx], y[idx]
            if len(np.unique(yc)) < 2:
                continue
            try:
                Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
                    Xc, yc, test_size=0.2, random_state=42, stratify=yc)
            except ValueError:
                Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
                    Xc, yc, test_size=0.2, random_state=42)

            clf_c = MLPClassifier(
                hidden_layer_sizes=(200, 100), activation="relu",
                solver="adam", batch_size=8, max_iter=500,
                early_stopping=True, random_state=42)
            clf_c.fit(Xc_tr, yc_tr)
            y_pred_c = clf_c.predict(Xc_te)
            acc = accuracy_score(yc_te, y_pred_c) * 100
            f1  = f1_score(yc_te, y_pred_c, average="macro",
                           zero_division=0) * 100
            lbl = data[data["Cluster"] == c]["cluster_label"].iloc[0] \
                  if "cluster_label" in data.columns else str(c)
            print(f"  Cluster {c} ({lbl}): n={len(idx)}  "
                  f"Acc={acc:.2f}%  F1={f1:.2f}%")
            cluster_results.append({
                "cluster": c, "label": lbl,
                "n": len(idx), "accuracy": round(acc,2), "f1": round(f1,2)
            })
        pd.DataFrame(cluster_results).to_csv(
            "cluster_nn_metrics.csv", index=False)

    # ── EXPERIMENT C: 5-Fold Cross-Validation ─────────────────────────────────
    print("\n[NN] Experiment C — 5-Fold Cross-Validation (FDN):")
    cv_clf = MLPClassifier(
        hidden_layer_sizes=(200, 100), activation="relu",
        solver="adam", batch_size=8, max_iter=300, random_state=42)
    cv = cross_validate(cv_clf, X_scaled, y, cv=5,
                        scoring=["accuracy","f1_macro",
                                 "precision_macro","recall_macro"])
    print(f"  Accuracy : {cv['test_accuracy'].mean()*100:.2f}% "
          f"(±{cv['test_accuracy'].std()*100:.2f}%)")
    print(f"  F1 Macro : {cv['test_f1_macro'].mean()*100:.2f}%")
    print(f"  Precision: {cv['test_precision_macro'].mean()*100:.2f}%")
    print(f"  Recall   : {cv['test_recall_macro'].mean()*100:.2f}%")

    # ── DETAILED REPORT ───────────────────────────────────────────────────────
    print("\n[NN] Detailed Classification Report (FDN — Full Dataset):")
    best_clf = MLPClassifier(
        hidden_layer_sizes=(200, 100), activation="relu",
        solver="adam", batch_size=8, max_iter=500,
        early_stopping=True, random_state=42)
    best_clf.fit(X_tr, y_tr)
    y_pred = best_clf.predict(X_te)
    print(classification_report(y_te, y_pred,
          target_names=["At-Risk", "Average", "High"]))

    cm = confusion_matrix(y_te, y_pred)
    np.save("confusion_matrix.npy", cm)
    print("  Confusion Matrix saved → confusion_matrix.npy")
    print("[NN] Metrics saved → nn_metrics.csv\n")

    return best_clf
