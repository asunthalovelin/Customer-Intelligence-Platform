"""
engine.py — Recommendation Engine Module
==========================================
Integrates predictions from the churn classifier and CLV regressor,
defines CLV value tiers, and assigns automated marketing recommendations
based on customer segment, churn risk, and value potential.
Also calculates financial revenue-at-risk for each customer.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    validate_config,
    PROCESSED_DIR,
    MODELS_DIR,
    CUSTOMER_SEGMENTED_FILE,
    CUSTOMER_INTELLIGENCE_FILE,
    CHURN_MODEL_FILE,
    CLV_MODEL_FILE,
    CLV_LOW_PERCENTILE,
    CLV_HIGH_PERCENTILE,
    CHURN_HIGH_RISK_THRESHOLD,
    CHURN_IMMEDIATE_THRESHOLD,
    CHURN_MODERATE_THRESHOLD
)

logger = logging.getLogger(__name__)

def build_recommendations() -> pd.DataFrame:
    validate_config()
    logger.info("Initializing Recommendation Engine...")

    # 1. Load segmented customer features
    segmented_path = PROCESSED_DIR / CUSTOMER_SEGMENTED_FILE
    logger.info(f"Loading segmented features from {segmented_path}...")
    df = pd.read_parquet(segmented_path)

    # 2. Load trained models
    churn_pipe = joblib.load(MODELS_DIR / CHURN_MODEL_FILE)
    clv_pipe = joblib.load(MODELS_DIR / CLV_MODEL_FILE)

    # 3. Generate predictions
    logger.info("Scoring customer database for Churn Risk and CLV...")
    
    # Preprocess feature columns for prediction (fill nulls to match training)
    # The models are Pipeline objects containing the preprocessors, so we pass raw features directly.
    # We must construct the feature sets exactly as used in trainer.py.
    from src.modeling.churn.trainer import NUMERIC_FEATURES as CHURN_NUMS, CATEGORICAL_FEATURES as CHURN_CATS
    from src.modeling.clv.trainer import NUMERIC_FEATURES as CLV_NUMS, CATEGORICAL_FEATURES as CLV_CATS
    
    churn_feats = CHURN_NUMS + CHURN_CATS
    clv_feats = CLV_NUMS + CLV_CATS
    
    # Align columns
    X_churn = df[[c for c in churn_feats if c in df.columns]].copy()
    X_clv = df[[c for c in clv_feats if c in df.columns]].copy()
    
    # Fill NAs
    for c in CHURN_NUMS:
        if c in X_churn.columns: X_churn[c] = X_churn[c].fillna(0)
    for c in CHURN_CATS:
        if c in X_churn.columns: X_churn[c] = X_churn[c].fillna("unknown").astype(str)
        
    for c in CLV_NUMS:
        if c in X_clv.columns: X_clv[c] = X_clv[c].fillna(0)
    for c in CLV_CATS:
        if c in X_clv.columns: X_clv[c] = X_clv[c].fillna("unknown").astype(str)

    df['churn_probability'] = churn_pipe.predict_proba(X_churn)[:, 1]
    
    predicted_clv_raw = clv_pipe.predict(X_clv)
    df['predicted_clv'] = np.clip(predicted_clv_raw, 0, None)  # CLV must be non-negative

    # 4. Define CLV Tiers based on percentiles
    clv_non_zero = df['predicted_clv'][df['predicted_clv'] > 0]
    if len(clv_non_zero) == 0:
        clv_non_zero = df['predicted_clv'] # fallback
        
    p33 = np.percentile(clv_non_zero, CLV_LOW_PERCENTILE)
    p67 = np.percentile(clv_non_zero, CLV_HIGH_PERCENTILE)
    logger.info(f"CLV Percentile thresholds: 33rd = R${p33:.2f} | 67th = R${p67:.2f}")

    # Assign CLV Tier
    conditions = [
        (df['predicted_clv'] <= p33),
        (df['predicted_clv'] > p33) & (df['predicted_clv'] <= p67),
        (df['predicted_clv'] > p67)
    ]
    choices = ['Low Value', 'Medium Value', 'High Value']
    df['clv_tier'] = np.select(conditions, choices, default='Low Value')

    # 5. Calculate Revenue at Risk: churn_probability * predicted_clv
    df['revenue_at_risk'] = df['churn_probability'] * df['predicted_clv']
    df['revenue_at_risk'] = df['revenue_at_risk'].round(2)

    # 6. Apply Rule-Based Action Mapping
    logger.info("Applying rule-based campaign routing...")
    df['recommended_action'] = df.apply(assign_action, axis=1)

    # Print summary statistics of recommendations
    action_counts = df['recommended_action'].value_counts()
    logger.info("Recommendation engine results summary:")
    for action, count in action_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  - {action}: {count:,} ({pct:.1f}%)")

    # 7. Save final integrated dataset
    out_path = PROCESSED_DIR / CUSTOMER_INTELLIGENCE_FILE
    df.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved final integrated dataset → {out_path}")
    logger.info(f"   Total rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

    return df

def assign_action(row: pd.Series) -> str:
    """
    Map customer segment, churn risk, and CLV to campaign routing labels.
    """
    segment = row['customer_segment']
    prob = row['churn_probability']
    clv_tier = row['clv_tier']

    # Rule 1: Extreme Churn Risk across any segment
    if prob > CHURN_IMMEDIATE_THRESHOLD:
        return "Immediate Outreach"

    # Rule 2: VIP campaigns
    if segment == "VIP Customers":
        if prob > CHURN_HIGH_RISK_THRESHOLD:
            return "Urgent Retention Campaign"
        else:
            return "Upsell Premium Products"

    # Rule 3: Loyal campaigns
    if segment == "Loyal Customers":
        if prob > CHURN_HIGH_RISK_THRESHOLD:
            return "Win-Back Campaign"
        else:
            return "Loyalty Incentive"

    # Rule 4: New customers onboarding
    if segment == "New Customers":
        return "Welcome Campaign"

    # Rule 5: At Risk recovery (Focus on Medium/High value)
    if segment == "At Risk Customers":
        if clv_tier in ["High Value", "Medium Value"]:
            return "Re-engagement Offer"
        else:
            return "Low Priority — Monitor"

    # Rule 6: Bargain hunters
    if segment == "Bargain Customers":
        return "Discount / Bundle Offer"

    # Rule 7: One-Time Buyers general fallback
    if segment == "One-Time Buyers":
        if prob > CHURN_MODERATE_THRESHOLD:
            if clv_tier == "High Value":
                return "Re-engagement Offer"
            else:
                return "Low Priority — Monitor"
        else:
            return "Welcome Campaign"

    # Standard Fallback
    return "Standard Marketing Newsletter"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    build_recommendations()
