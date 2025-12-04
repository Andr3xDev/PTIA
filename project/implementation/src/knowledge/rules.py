"""
Simple Rule-Based Expert System for Software Quality Assessment.
Uses clear threshold rules to classify code modules as PASS/WARN/FAIL.
"""

import pandas as pd

# Thresholds calibrated distributions
THRESHOLDS = {
    "complexity": 8,  # McCabe: >8 starts being complex
    "maintainability": 60,  # MI: <60 could improve
    "effort": 25000,  # Halstead: >25k is moderate
    "size": 150,  # LOC: >150 is medium
    "doc_density": 0.12,  # <0.12 needs more docs
}


def classify_module(row: pd.Series) -> dict:
    """
    Classify a single module using simple threshold rules.

    Rules:
    - FAIL: 3+ metrics are risky OR complexity > 20 (obvious technical debt)
    - WARN: 1-2 metrics are risky (gray zone - needs AI analysis)
    - PASS: All metrics OK (clean code)
    """
    risks = []

    # Check each metric against threshold
    if row["complexity"] > THRESHOLDS["complexity"]:
        risks.append("complexity")
    if row["maintainability"] < THRESHOLDS["maintainability"]:
        risks.append("maintainability")
    if row["effort"] > THRESHOLDS["effort"]:
        risks.append("effort")
    if row["size"] > THRESHOLDS["size"]:
        risks.append("size")
    if row["doc_density"] < THRESHOLDS["doc_density"]:
        risks.append("doc_density")

    # Determine status
    if len(risks) >= 2 or row["complexity"] > 20:
        status = "FAIL"
    elif len(risks) >= 1:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "risk_count": len(risks),
        "risks": ", ".join(risks) if risks else "none",
    }


def evaluate_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply rules to entire dataset.
    Returns DataFrame with status, risk_count, and risks columns.
    """
    results = df.apply(classify_module, axis=1, result_type="expand")
    result_df = pd.concat([df, results], axis=1)

    if verbose:
        counts = result_df["status"].value_counts()
        total = len(df)
        print(f"Quality Assessment Results ({total} modules):")
        print(
            f"  PASS: {counts.get('PASS', 0)} ({counts.get('PASS', 0)/total*100:.1f}%)"
        )
        print(
            f"  WARN: {counts.get('WARN', 0)} ({counts.get('WARN', 0)/total*100:.1f}%)"
        )
        print(
            f"  FAIL: {counts.get('FAIL', 0)} ({counts.get('FAIL', 0)/total*100:.1f}%)"
        )

    return result_df


def get_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics from evaluated dataset."""
    return {
        "total": len(df),
        "pass_rate": (df["status"] == "PASS").mean() * 100,
        "warn_rate": (df["status"] == "WARN").mean() * 100,
        "fail_rate": (df["status"] == "FAIL").mean() * 100,
        "avg_risks": df["risk_count"].mean(),
    }
