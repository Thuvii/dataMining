import json
from textwrap import dedent
# Paste your notebook JSON here

   
nb = {
    "cells": [
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": dedent(r"""
    # Step‑by‑Step: Which Early‑Game Features Are Most Predictive of Winning?
    
    This notebook combines the ideas from your **`LogisticRegression.ipynb`** and **`Importance_SHAP.ipynb`**:
    - **Logistic Regression (with scaling)** for **direction + odds ratios**
    - **Tree model + SHAP** for **nonlinear, interaction-aware importance**
    
    We'll proceed in small, clear steps.
    
    **Files expected:**
    - Dataset CSV: `high_diamond_ranked_10min.csv` (same folder as this notebook)
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 0 — Setup"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    import os, math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.ensemble import RandomForestClassifier
    
    print("Libraries loaded.")
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 1 — Load data"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    DATA_PATH = "high_diamond_ranked_10min.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Put 'high_diamond_ranked_10min.csv' next to this notebook, or update DATA_PATH.")
    
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    display(df.head())
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 2 — Target and basic checks"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    assert "blueWins" in df.columns, "Expected 'blueWins' column (1=blue win, 0=red win)."
    y = df["blueWins"].astype(int)
    print("Blue win rate:", y.mean())
    
    # Keep numeric features only
    X_num = df.drop(columns=["blueWins"]).select_dtypes(include=[np.number]).copy()
    print("Numeric feature count:", X_num.shape[1])
    
    # Quick missingness check
    missing = X_num.isna().mean()
    print("Columns with missing values:")
    display(missing[missing>0].sort_values(ascending=False))
    X_num = X_num.fillna(0)
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 3 — Create Blue–Red difference features (symmetry, less redundancy)"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    # We auto-pair columns like 'blueGold' and 'redGold' and create diff = blue - red.
    blue_cols = [c for c in X_num.columns if c.startswith("blue")]
    red_cols = [c for c in X_num.columns if c.startswith("red")]
    
    def red_partner(blue_name):  # 'blueGold' -> 'redGold'
        return "red" + blue_name[len("blue"):]
    
    diffs = {}
    for b in blue_cols:
        r = red_partner(b)
        if r in red_cols and np.issubdtype(X_num[b].dtype, np.number) and np.issubdtype(X_num[r].dtype, np.number):
            diffs["diff" + b[len("blue"):]] = X_num[b] - X_num[r]
    
    X_diff = pd.DataFrame(diffs)
    print("Created diff features:", X_diff.shape[1])
    display(X_diff.head())
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 4 — Train/Test split (stratified)"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    X_train, X_test, y_train, y_test = train_test_split(X_diff, y, test_size=0.25, random_state=42, stratify=y)
    print("Train size:", X_train.shape, " Test size:", X_test.shape)
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 5 — Logistic Regression (with scaling)"
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": dedent(r"""
    **Why scaling?** Logistic Regression coefficients are sensitive to feature scale.  
    Standardizing to mean 0, std 1 makes coefficients comparable: a 1‑unit change is **1 standard deviation**.
    """).strip()
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    logreg = LogisticRegression(max_iter=300, n_jobs=None)
    logreg.fit(X_train_sc, y_train)
    
    y_prob = logreg.predict_proba(X_test_sc)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print("Logistic Regression — Accuracy:", f"{acc:.3f}", " AUC:", f"{auc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))
    
    # Coefficients -> odds ratios per 1 SD
    coef = pd.Series(logreg.coef_[0], index=X_train.columns)
    odds = np.exp(coef)
    lr_importance = pd.DataFrame({"coef_std": coef, "odds_ratio_per_SD": odds}).sort_values("coef_std", key=lambda s: s.abs(), ascending=False)
    display(lr_importance.head(15))
    
    # Plot top coefficients
    top15 = lr_importance.head(15).copy()
    plt.figure()
    top15["coef_std"].plot(kind="bar")
    plt.title("Logistic Regression (standardized) — Top Coefficients")
    plt.ylabel("Coefficient (per 1 SD)")
    plt.xticks(rotation=90)
    plt.show()
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 6 — Random Forest (nonlinear baseline)"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)[:,1]
        rf_pred = (rf_prob >= 0.5).astype(int)
        
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_auc = roc_auc_score(y_test, rf_prob)
        print("Random Forest — Accuracy:", f"{rf_acc:.3f}", " AUC:", f"{rf_auc:.3f}")
        
        rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        display(rf_importance.head(15))
        
        plt.figure()
        rf_importance.head(15).plot(kind="bar")
        plt.title("Random Forest — Top Feature Importances")
        plt.ylabel("Importance")
        plt.xticks(rotation=90)
        plt.show()
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 7 — SHAP for Interpretability (tree model)"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
        try:
            import shap
            # TreeExplainer works well for random forest
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)
            
            # Global importance: mean(|SHAP|) per feature
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_abs_mean = np.abs(sv).mean(axis=0)
            shap_importance = pd.Series(shap_abs_mean, index=X_train.columns).sort_values(ascending=False)
            print("Top 15 SHAP (mean |value|):")
            display(shap_importance.head(15))
            
            # Summary bar plot
            shap.summary_plot(sv, X_test, plot_type="bar")
        except Exception as e:
            print("SHAP not available or failed to plot:", e)
            shap_importance = pd.Series(dtype=float)
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": "## Step 8 — Consensus ranking"
    },
    {
    "cell_type": "code",
    "metadata": {},
    "source": dedent(r"""
    # Build a consensus by averaging normalized ranks across LR, RF, SHAP (available ones)
    frames = []
    # Logistic: absolute coefficient magnitude
    if "lr_importance" in locals():
        r1 = lr_importance.copy()
        r1["score"] = r1["coef_std"].abs()
        r1 = r1[["score"]].rename(columns={"score":"LR"})
        frames.append(r1)
    # RF
    if "rf_importance" in locals():
        r2 = rf_importance.to_frame(name="RF")
        frames.append(r2)
    # SHAP
    if "shap_importance" in locals() and len(shap_importance)>0:
        r3 = shap_importance.to_frame(name="SHAP")
        frames.append(r3)
    
    if frames:
        combo = pd.concat(frames, axis=1)
        # Normalize each column to [0,1] and average
        for col in combo.columns:
            m, M = combo[col].min(), combo[col].max()
            if M > m:
                combo[col] = (combo[col]-m)/(M-m)
        combo["consensus"] = combo.mean(axis=1)
        combo_sorted = combo.sort_values("consensus", ascending=False)
        print("Consensus top 15 features:")
        display(combo_sorted.head(15))
        
        # Save for reporting
        out_path = "feature_importance_consensus.csv"
        combo_sorted.to_csv(out_path)
        print("Saved:", out_path)
        
        plt.figure()
        combo_sorted["consensus"].head(15).plot(kind="bar")
        plt.title("Consensus Importance — Top 15")
        plt.ylabel("Normalized consensus score")
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("No importance objects available to build consensus.")
    """).strip()
    },
    {
    "cell_type": "markdown",
    "metadata": {},
    "source": dedent(r"""
    ## How to interpret the outputs
    
    - **Logistic Regression (scaled):** coefficients are per **1 standard deviation**.  
        - Positive ⇒ increasing the feature raises the odds of Blue winning.  
        - Negative ⇒ increasing the feature lowers the odds.  
        - `odds_ratio_per_SD = exp(coef)` tells how many times the odds change per 1 SD increase.
    
    - **Random Forest importance:** how much a feature reduces impurity. Good for **nonlinear** signals.
    
    - **SHAP (tree):** mean absolute SHAP value = average impact magnitude on the model's prediction.  
        - Summary bar plot ranks features globally.  
        - (Optional) `shap.summary_plot(sv, X_test)` gives a dot plot to see how high/low feature values push predictions.
    
    - **Consensus ranking:** averages normalized importances to highlight features that are consistently important across methods.
    """).strip()
    }
    ],
    "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}



path = "Step_by_Step_Feature_Importance.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook saved to:", path)
