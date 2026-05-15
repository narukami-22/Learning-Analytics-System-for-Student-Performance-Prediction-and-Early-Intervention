"""
intervention_system.py — Early Intervention Engine
====================================================
Fixes from original:
  - Only 2 rules (absences + studytime) → now 7 multi-factor rules
  - Ignored cluster labels entirely → now cluster-aware routing
  - No urgency levels → added High / Medium / Low urgency
  - No risk factor list → now tracks which factors triggered
  - Added at-risk student summary report
"""

import pandas as pd


# ── RISK FACTOR DETECTION ─────────────────────────────────────────────────────

def _get_risk_factors(row):
    """Detect and return all risk factors for a student."""
    risks = []
    if row.get("failures", 0) >= 2:
        risks.append("repeated_failures")
    if row.get("absences", 0) > 15:
        risks.append("high_absences")
    if row.get("studytime", 3) <= 1:
        risks.append("low_study_time")
    if row.get("Dalc", 1) >= 4 or row.get("Walc", 1) >= 4:
        risks.append("high_alcohol_use")
    if row.get("G1", 10) < 8 and row.get("G2", 10) < 8:
        risks.append("consistently_low_grades")
    if row.get("health", 3) <= 2:
        risks.append("poor_health")
    if row.get("Medu", 2) <= 1 and row.get("Fedu", 2) <= 1:
        risks.append("low_parental_education")
    return risks


# ── INTERVENTION ASSIGNMENT ───────────────────────────────────────────────────

def _assign_interventions(risk_factors, cluster_label):
    """Map risk factors + cluster label to targeted interventions."""
    interventions = []

    # Cluster-level intervention
    if cluster_label == "At-Risk":
        interventions.append("Priority academic counseling")

    # Factor-specific interventions
    if "repeated_failures" in risk_factors:
        interventions.append("Remedial tutoring program")
    if "high_absences" in risk_factors:
        interventions.append("Attendance counseling + parent contact")
    if "low_study_time" in risk_factors:
        interventions.append("Study skills workshop")
    if "high_alcohol_use" in risk_factors:
        interventions.append("Wellness & substance awareness program")
    if "consistently_low_grades" in risk_factors:
        interventions.append("Early exam preparation support")
    if "poor_health" in risk_factors:
        interventions.append("Health & wellbeing referral")
    if "low_parental_education" in risk_factors:
        interventions.append("Family engagement program")

    if not interventions:
        interventions.append("Standard monitoring")

    return " | ".join(interventions)


# ── URGENCY LEVEL ─────────────────────────────────────────────────────────────

def _urgency_level(risk_factors, cluster_label):
    """Assign urgency tier: High / Medium / Low."""
    score = len(risk_factors)
    if cluster_label == "At-Risk" or score >= 3:
        return "High"
    elif score >= 1:
        return "Medium"
    else:
        return "Low"


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def apply_intervention(data):
    """
    Apply multi-factor, cluster-aware intervention logic to every student.

    Adds columns:
      risk_factors      — list of detected risk factors
      risk_factor_count — number of risk factors
      intervention      — recommended action(s)
      urgency           — High / Medium / Low
    """
    if "cluster_label" not in data.columns:
        data = data.copy()
        data["cluster_label"] = "Unknown"

    data["risk_factors"] = data.apply(_get_risk_factors, axis=1)
    data["risk_factor_count"] = data["risk_factors"].apply(len)

    data["intervention"] = data.apply(
        lambda row: _assign_interventions(
            row["risk_factors"], row["cluster_label"]), axis=1)

    data["urgency"] = data.apply(
        lambda row: _urgency_level(
            row["risk_factors"], row["cluster_label"]), axis=1)

    # Summary
    print("\n[Intervention] Urgency Distribution:")
    print(data["urgency"].value_counts().to_string())
    print("\n[Intervention] Top Interventions Assigned:")
    all_itv = data["intervention"].str.split(" | ").explode()
    print(all_itv.value_counts().head(8).to_string())

    return data


# ── AT-RISK REPORT ────────────────────────────────────────────────────────────

def print_at_risk_report(data, top_n=10):
    """Print the highest-risk students with their interventions."""
    at_risk = (data[data["urgency"] == "High"]
               .sort_values("risk_factor_count", ascending=False)
               .head(top_n))

    print(f"\n[Intervention] Top {top_n} Highest-Risk Students:")
    display_cols = ["student_id", "G3", "absences", "failures",
                    "cluster_label", "urgency", "intervention"]
    display_cols = [c for c in display_cols if c in at_risk.columns]
    print(at_risk[display_cols].to_string(index=False))
