"""
revenue.py — Revenue Feature Engineering
==========================================
Financial features that capture the economic value and spending pattern
of each customer.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revenue-based features.

    Features created:
        total_revenue      : Sum of all payment values across all orders
        avg_order_value    : Average payment value per order
        max_order_value    : Highest single order value (proxy for high-value intent)
        avg_freight_value  : Average freight cost per order
    """
    out = df.copy()

    # Already present from joiner; ensure clean types
    out["total_revenue"] = out["total_revenue"].fillna(0).round(2)
    out["avg_order_value"] = out["avg_order_value"].fillna(0).round(2)
    out["max_order_value"] = out["max_order_value"].fillna(0).round(2)
    out["avg_freight_value"] = out["avg_freight_value"].fillna(0).round(2)

    logger.info(
        f"Revenue features computed | "
        f"Total revenue range: R${out['total_revenue'].min():.2f} – "
        f"R${out['total_revenue'].max():.2f}"
    )
    return out
