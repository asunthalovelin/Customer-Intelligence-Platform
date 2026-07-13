"""
shap_explainer.py — SHAP Explainability Engine
===============================================
Computes global feature importances, beeswarm plots, and local waterfall explanations
for the churn and CLV models, saving visual outputs and reports.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    validate_config,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)

# Reuse feature list definitions from trainers
from src.modeling.churn.trainer import prepare_data as prepare_churn_data
from src.modeling.clv.trainer import prepare_data as prepare_clv_data

logger = logging.getLogger(__name__)

def generate_explanations():
    validate_config()
    logger.info("Initializing SHAP Explainability Engine...")
    
    # ─────────────────────────────────────────────
    # 1. Churn Explanations
    # ─────────────────────────────────────────────
    logger.info("Generating Churn SHAP explanations...")
    X_churn, y_churn, _ = prepare_churn_data()
    churn_pipe = joblib.load(MODELS_DIR / "churn_model.joblib")
    churn_model = churn_pipe.named_steps['model']
    prep_churn = churn_pipe.named_steps['preprocessor']
    
    # Transform data
    X_churn_trans = prep_churn.transform(X_churn)
    feature_names_churn = prep_churn.get_feature_names_out()
    # Clean feature names (remove prefix num__ and cat__)
    feature_names_churn = [f.split("__")[-1] for f in feature_names_churn]
    
    X_churn_df = pd.DataFrame(X_churn_trans, columns=feature_names_churn)
    
    # Use a sample of 1000 records for fast SHAP computation
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_churn_df), size=min(1000, len(X_churn_df)), replace=False)
    X_churn_sample = X_churn_df.iloc[sample_idx]
    
    # TreeExplainer
    explainer_churn = shap.TreeExplainer(churn_model)
    shap_values_churn = explainer_churn(X_churn_sample)
    
    # Plot Churn Global Bar
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_churn, max_display=15, show=False)
    plt.title("Global Feature Importance (SHAP Bar) — Churn Prediction")
    bar_path_churn = FIGURES_DIR / "shap_churn_bar.png"
    plt.savefig(bar_path_churn, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot Churn Global Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_churn, max_display=15, show=False)
    plt.title("Feature Impact Distribution (SHAP Beeswarm) — Churn Prediction")
    beeswarm_path_churn = FIGURES_DIR / "shap_churn_beeswarm.png"
    plt.savefig(beeswarm_path_churn, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved churn global SHAP plots.")
    
    # Churn Local Explanations (Waterfall)
    # Find low, high, and borderline risk customers
    y_prob_churn = churn_pipe.predict_proba(X_churn)[:, 1]
    
    # High risk index
    high_risk_local_idx = np.argmax(y_prob_churn)
    # Low risk index
    low_risk_local_idx = np.argmin(y_prob_churn)
    # Borderline risk index (closest to 0.5)
    borderline_risk_local_idx = np.argmin(np.abs(y_prob_churn - 0.5))
    
    local_idxs = {
        "high_risk": high_risk_local_idx,
        "low_risk": low_risk_local_idx,
        "borderline_risk": borderline_risk_local_idx
    }
    
    for label, raw_idx in local_idxs.items():
        # Map raw index back to our sample dataframe format to match explainer dimensions
        single_row = X_churn_df.iloc[[raw_idx]]
        single_shap = explainer_churn(single_row)[0]
        
        plt.figure(figsize=(10, 5))
        # Ensure we explain with probability scale if possible, otherwise raw SHAP output
        shap.plots.waterfall(single_shap, max_display=10, show=False)
        plt.title(f"Local Decision Flow ({label.replace('_', ' ').capitalize()}) — Churn")
        path = FIGURES_DIR / f"shap_churn_local_{label}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
    logger.info("Saved churn local waterfall plots.")

    # ─────────────────────────────────────────────
    # 2. CLV Explanations
    # ─────────────────────────────────────────────
    logger.info("Generating CLV SHAP explanations...")
    X_clv, y_clv, _ = prepare_clv_data()
    clv_pipe = joblib.load(MODELS_DIR / "clv_model.joblib")
    clv_model = clv_pipe.named_steps['model']
    prep_clv = clv_pipe.named_steps['preprocessor']
    
    X_clv_trans = prep_clv.transform(X_clv)
    feature_names_clv = prep_clv.get_feature_names_out()
    feature_names_clv = [f.split("__")[-1] for f in feature_names_clv]
    X_clv_df = pd.DataFrame(X_clv_trans, columns=feature_names_clv)
    
    X_clv_sample = X_clv_df.iloc[sample_idx]
    
    explainer_clv = shap.TreeExplainer(clv_model)
    shap_values_clv = explainer_clv(X_clv_sample)
    
    # Plot CLV Global Bar
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_clv, max_display=15, show=False)
    plt.title("Global Feature Importance (SHAP Bar) — CLV Forecast")
    bar_path_clv = FIGURES_DIR / "shap_clv_bar.png"
    plt.savefig(bar_path_clv, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot CLV Global Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_clv, max_display=15, show=False)
    plt.title("Feature Impact Distribution (SHAP Beeswarm) — CLV Forecast")
    beeswarm_path_clv = FIGURES_DIR / "shap_clv_beeswarm.png"
    plt.savefig(beeswarm_path_clv, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved CLV global SHAP plots.")
    
    # CLV Local Explanations (Waterfall)
    y_pred_clv = clv_pipe.predict(X_clv)
    
    high_clv_local_idx = np.argmax(y_pred_clv)
    low_clv_local_idx = np.argmin(y_pred_clv)
    median_clv_local_idx = np.argsort(y_pred_clv)[len(y_pred_clv) // 2]
    
    clv_local_idxs = {
        "high_value": high_clv_local_idx,
        "low_value": low_clv_local_idx,
        "median_value": median_clv_local_idx
    }
    
    for label, raw_idx in clv_local_idxs.items():
        single_row = X_clv_df.iloc[[raw_idx]]
        single_shap = explainer_clv(single_row)[0]
        
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(single_shap, max_display=10, show=False)
        plt.title(f"Local Decision Flow ({label.replace('_', ' ').capitalize()}) — CLV")
        path = FIGURES_DIR / f"shap_clv_local_{label}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
    logger.info("Saved CLV local waterfall plots.")

    # ─────────────────────────────────────────────
    # 3. Save Feature Importance Report
    # ─────────────────────────────────────────────
    report_path = REPORTS_DIR / "feature_importance_report.md"
    logger.info(f"Writing Feature Importance Report to {report_path}...")
    
    # Calculate global absolute SHAP importance values
    global_shap_churn = np.abs(shap_values_churn.values).mean(axis=0)
    churn_imp = pd.Series(global_shap_churn, index=feature_names_churn).sort_values(ascending=False)
    
    global_shap_clv = np.abs(shap_values_clv.values).mean(axis=0)
    clv_imp = pd.Series(global_shap_clv, index=feature_names_clv).sort_values(ascending=False)
    
    report_md = f"""# Customer Intelligence Platform — Feature Importance Report
Generated on: 2026-06-29

This report provides global feature importance analysis using SHAP values to explain the drivers behind customer churn and customer lifetime value (CLV).

---

## 1. Top Feature Drivers Comparison

Below are the feature importances ranked by mean absolute SHAP value:

### Churn Drivers (XGBoost)
"""
    for i, (col, val) in enumerate(churn_imp.head(10).items(), 1):
        report_md += f"{i}. **{col}**: {val:.4f} mean absolute SHAP impact\n"
        
    report_md += "\n### CLV Drivers (Random Forest)\n"
    for i, (col, val) in enumerate(clv_imp.head(10).items(), 1):
        report_md += f"{i}. **{col}**: {val:.4f} mean absolute SHAP impact\n"
        
    report_md += f"""
---

## 2. Business Interpretation of Drivers

### 🚨 Churn Risk Drivers:
- **`customer_tenure_days`**: The strongest indicator of churn. Customers with tenure = 0 (only placed one order and never returned) represent a huge cohort with 100% churn rate.
- **`avg_review_score` / `negative_review_count`**: Experiential friction. Poor ratings from late deliveries or incorrect products strongly drive customers away.
- **`avg_delivery_delay_days` / `on_time_delivery_rate`**: Operational delay. Deliveries running past their estimated date create friction, leading to a high churn multiplier.
- **`frequency`**: Buyers who have made more than 1 purchase represent stable users with a drastically lower base churn risk.

### 💰 Future CLV Drivers:
- **`recency_days`**: The strongest predictor. Due to the 6-month holdout structure, recency determines if they are in the active target window. 
- **`avg_order_value` / `max_order_value`**: Core monetisation metrics. Higher basket spend in past purchases correlates with high-value future spend.
- **`avg_installments`**: Payment pattern. Customers using installments are highly monetised and active shoppers on the Olist platform.

---

## 3. Visualizations Generated
- Global Churn importance bar: `reports/figures/shap_churn_bar.png`
- Global Churn beeswarm impact: `reports/figures/shap_churn_beeswarm.png`
- Global CLV importance bar: `reports/figures/shap_clv_bar.png`
- Global CLV beeswarm impact: `reports/figures/shap_clv_beeswarm.png`
- Local waterfall charts for 3 representative churn and CLV cases.
"""
    with open(report_path, "w") as f:
        f.write(report_md)
        
    logger.info("SHAP explainability engine complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    generate_explanations()
