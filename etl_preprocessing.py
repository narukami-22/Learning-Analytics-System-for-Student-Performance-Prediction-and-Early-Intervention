"""
etl_preprocessing.py — ETL Pipeline & Dimensional Modeling
============================================================
Fixes from original:
  - Loads BOTH Math and Portuguese datasets (not just Math)
  - Fixed LabelEncoder bug (one encoder per column, not shared)
  - Added feature engineering (performance_class, avg_grade)
  - Added star schema dimensional modeling (SQLite)
  - Added subject tagging and student_id
  - Saves transformed CSV for downstream phases
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.preprocessing import LabelEncoder


def preprocess_data():
    # ── EXTRACT ──────────────────────────────────────────────────────────────
    mat_path = "dataset/student-mat.csv"
    por_path = "dataset/student-por.csv"

    # Fallback paths if dataset folder not present
    if not os.path.exists(mat_path):
        mat_path = "student-mat.csv"
    if not os.path.exists(por_path):
        por_path = "student-por.csv"

    df_mat = pd.read_csv(mat_path, sep=";")
    df_por = pd.read_csv(por_path, sep=";")

    df_mat["subject"] = "Math"
    df_por["subject"] = "Portuguese"

    data = pd.concat([df_mat, df_por], ignore_index=True)
    data["student_id"] = range(1, len(data) + 1)

    print(f"[ETL] Loaded: Math={len(df_mat)}, Portuguese={len(df_por)}, "
          f"Total={len(data)}")

    # ── TRANSFORM ─────────────────────────────────────────────────────────────
    # Fix grade column types
    for col in ["G1", "G2", "G3"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

    # Drop rows with missing values
    before = len(data)
    data = data.dropna()
    print(f"[ETL] Dropped {before - len(data)} rows with nulls. "
          f"Remaining: {len(data)}")

    # Encode binary yes/no columns
    binary_cols = ["schoolsup", "famsup", "paid", "activities",
                   "nursery", "higher", "internet", "romantic"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({"yes": 1, "no": 0})

    # Encode remaining categorical columns
    # FIX: use a separate LabelEncoder instance per column (original bug)
    cat_cols = [c for c in data.columns
                if data[c].dtype == "object" and c not in ["subject"]]
    for col in cat_cols:
        le = LabelEncoder()
        data[col + "_enc"] = le.fit_transform(data[col].astype(str))

    # Simple manual encodings
    if "sex" in data.columns:
        data["sex_enc"] = data["sex"].map({"M": 1, "F": 0})
    if "address" in data.columns:
        data["address_enc"] = data["address"].map({"U": 1, "R": 0})
    if "famsize" in data.columns:
        data["famsize_enc"] = data["famsize"].map({"GT3": 1, "LE3": 0})
    if "Pstatus" in data.columns:
        data["Pstatus_enc"] = data["Pstatus"].map({"T": 1, "A": 0})

    # Derived features
    data["avg_grade"] = (data["G1"] + data["G2"] + data["G3"]) / 3

    def grade_label(g3):
        if g3 >= 15:   return "High"
        elif g3 >= 10: return "Average"
        else:          return "At-Risk"

    data["performance_class"] = data["G3"].apply(grade_label)

    print(f"[ETL] Performance class distribution:")
    print(data["performance_class"].value_counts().to_string())

    # ── LOAD — Star Schema (SQLite) ───────────────────────────────────────────
    conn = sqlite3.connect("student_dw.db")

    # Fact table
    fact_cols = ["student_id", "subject", "G1", "G2", "G3",
                 "avg_grade", "performance_class", "absences", "failures"]
    data[fact_cols].to_sql("FACT_STUDENT_PERFORMANCE", conn,
                           if_exists="replace", index=False)

    # Dimension: Student demographics
    dim_s = ["student_id", "sex", "age", "address", "school",
             "subject", "sex_enc", "address_enc"]
    data[[c for c in dim_s if c in data.columns]].to_sql(
        "DIM_STUDENT", conn, if_exists="replace", index=False)

    # Dimension: Family background
    dim_f = ["student_id", "famsize", "Pstatus", "Medu", "Fedu",
             "Mjob", "Fjob", "guardian", "famsup", "famrel"]
    data[[c for c in dim_f if c in data.columns]].to_sql(
        "DIM_FAMILY", conn, if_exists="replace", index=False)

    # Dimension: School environment
    dim_sc = ["student_id", "studytime", "failures", "schoolsup",
              "paid", "activities", "higher", "internet", "traveltime"]
    data[[c for c in dim_sc if c in data.columns]].to_sql(
        "DIM_SCHOOL", conn, if_exists="replace", index=False)

    # Dimension: Lifestyle behaviour
    dim_b = ["student_id", "freetime", "goout", "Dalc", "Walc", "health", "romantic"]
    data[[c for c in dim_b if c in data.columns]].to_sql(
        "DIM_BEHAVIOR", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print("[ETL] Star schema written to student_dw.db "
          "(1 FACT + 4 DIMENSION tables)")

    # Save transformed CSV for all downstream modules
    data.to_csv("cleaned_student_data.csv", index=False)
    print("[ETL] Saved → cleaned_student_data.csv\n")

    return data
