"""
behavior.py — Purchase Behavior Feature Engineering
=====================================================
Captures the temporal and transactional patterns of customer purchasing.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute purchase behavior features.

    Features created:
        total_orders               : Total distinct orders placed
        customer_tenure_days       : Days between first and last purchase
        avg_days_between_orders    : Average gap between consecutive orders
        repeat_purchase_rate       : 1 if customer ordered more than once, else 0
    """
    out = df.copy()

    # total_orders already exists from joiner
    out["total_orders"] = out["total_orders"].fillna(1).astype(int)

    # Tenure: first to last purchase
    out["customer_tenure_days"] = (
        out["last_purchase_date"] - out["first_purchase_date"]
    ).dt.total_seconds() / 86400
    out["customer_tenure_days"] = out["customer_tenure_days"].clip(lower=0).round(0)

    # Average days between orders
    # For single-order customers: set to 0 (undefined gap)
    out["avg_days_between_orders"] = np.where(
        out["total_orders"] > 1,
        out["customer_tenure_days"] / (out["total_orders"] - 1),
        0.0,
    )
    out["avg_days_between_orders"] = out["avg_days_between_orders"].round(1)

    # Repeat purchase rate (binary): 1 if more than one order
    out["repeat_purchase_rate"] = (out["total_orders"] > 1).astype(int)

    logger.info(
        f"Behavior features computed | "
        f"Repeat buyers: {out['repeat_purchase_rate'].sum():,} "
        f"({out['repeat_purchase_rate'].mean()*100:.1f}%)"
    )
    return out
