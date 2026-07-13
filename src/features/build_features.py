"""
build_features.py — Build Customer Features Pipeline
======================================================
Combines RFM, revenue, behavioral, review, payment, and delivery modules
to produce the final customer_features.parquet dataset.
"""

import sys
import logging
from pathlib import Path

# Add project root to path for local src imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd

from src.config import validate_config, PROCESSED_DIR, CUSTOMER_BASE_FILE, CUSTOMER_FEATURES_FILE
from src.features.rfm import compute_rfm
from src.features.revenue import compute_revenue_features
from src.features.behavior import compute_behavior_features
from src.features.reviews import compute_review_features
from src.features.payments import compute_payment_features
from src.features.delivery import compute_delivery_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("build_features")

def run_feature_pipeline() -> pd.DataFrame:
    validate_config()
    
    # 1. Load customer base parquet
    base_path = PROCESSED_DIR / CUSTOMER_BASE_FILE
    logger.info(f"Loading base dataset from {base_path}...")
    df = pd.read_parquet(base_path)
    
    # 2. Sequential execution of feature modules
    logger.info("Executing RFM feature calculations...")
    df = compute_rfm(df)
    
    logger.info("Executing Revenue feature calculations...")
    df = compute_revenue_features(df)
    
    logger.info("Executing Behavior feature calculations...")
    df = compute_behavior_features(df)
    
    logger.info("Executing Review feature calculations...")
    df = compute_review_features(df)
    
    logger.info("Executing Payment feature calculations...")
    df = compute_payment_features(df)
    
    logger.info("Executing Delivery feature calculations...")
    df = compute_delivery_features(df)
    
    # 3. Save final features table
    out_path = PROCESSED_DIR / CUSTOMER_FEATURES_FILE
    df.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved customer features parquet to {out_path}")
    logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df

if __name__ == "__main__":
    run_feature_pipeline()
