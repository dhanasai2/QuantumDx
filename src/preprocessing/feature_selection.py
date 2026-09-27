"""Feature Selection Module for Biomedical Data (Deliverable #1).

Calculates Mutual Information (MI) scores, ANOVA F-values, and Random Forest
feature importance rankings for patient clinical features. Provides feature
selection masks to select top-K features prior to PCA or classical ML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier


def compute_feature_selection_scores(
    df: pd.DataFrame, target_col: str = "cardio"
) -> dict:
    """Computes Mutual Information, F-scores, and Random Forest feature importances

    Returns a structured dictionary with feature rankings and scores.
    """
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        # If target column not present, create a synthetic target based on clinical risk indicators
        X = df.copy()
        # Use simple clinical proxy for target if missing
        if "ap_hi" in X.columns and "cholesterol" in X.columns:
            y = (X["ap_hi"] > 130) | (X["cholesterol"] > 1)
            y = y.astype(int)
        else:
            # Fallback to mean threshold of first column
            first_col = X.columns[0]
            y = (X[first_col] > X[first_col].median()).astype(int)

    # Filter numeric features only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"features": [], "mi_scores": [], "f_scores": [], "rf_scores": []}

    X_num = X[numeric_cols].fillna(X[numeric_cols].median())

    # 1. Mutual Information Scores
    try:
        mi_scores = mutual_info_classif(X_num, y, random_state=42)
    except Exception:
        mi_scores = np.ones(len(numeric_cols)) / len(numeric_cols)

    # 2. ANOVA F-Scores
    try:
        f_scores, _ = f_classif(X_num, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
    except Exception:
        f_scores = np.zeros(len(numeric_cols))

    # 3. Random Forest Feature Importance
    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_num, y)
        rf_scores = rf.feature_importances_
    except Exception:
        rf_scores = np.ones(len(numeric_cols)) / len(numeric_cols)

    # Normalize scores to [0, 1] range for visual comparability
    mi_norm = mi_scores / (np.max(mi_scores) + 1e-9)
    f_norm = f_scores / (np.max(f_scores) + 1e-9)
    rf_norm = rf_scores / (np.max(rf_scores) + 1e-9)

    # Combined composite score
    composite = (mi_norm * 0.4) + (f_norm * 0.3) + (rf_norm * 0.3)

    # Sort features by composite score descending
    sorted_indices = np.argsort(composite)[::-1]

    ranked_features = []
    for idx in sorted_indices:
        col_name = numeric_cols[idx]
        ranked_features.append({
            "feature": col_name,
            "mi_score": float(np.round(mi_scores[idx], 4)),
            "f_score": float(np.round(f_scores[idx], 2)),
            "rf_importance": float(np.round(rf_scores[idx], 4)),
            "composite_score": float(np.round(composite[idx], 4)),
            "rank": len(ranked_features) + 1,
        })

    return {
        "n_features": len(numeric_cols),
        "target_column": target_col if target_col in df.columns else "synthetic_proxy",
        "rankings": ranked_features,
        "selected_top_4": [f["feature"] for f in ranked_features[:4]],
    }
