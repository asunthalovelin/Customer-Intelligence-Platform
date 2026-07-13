"""
delivery.py — Delivery Feature Engineering
============================================
Delivery experience directly impacts customer satisfaction and repurchase intent.
Late deliveries are a leading driver of negative reviews and churn.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delivery-related features.

    Features created:
        avg_delivery_delay_days : Mean delivery delay in days
                                  (positive = late, negative = early)
        delayed_order_count     : Number of orders delivered after estimated date
        on_time_delivery_rate   : Fraction of orders delivered on time (0–1)
    """
    out = df.copy()

    out["avg_delivery_delay_days"] = (
        out["avg_delivery_delay_days"].fillna(0).round(2)
    )
    out["delayed_order_count"] = (
        out["delayed_order_count"].fillna(0).astype(int)
    )
    out["on_time_delivery_rate"] = (
        out["on_time_delivery_rate"].fillna(1.0).clip(0, 1).round(3)
    )

    logger.info(
        f"Delivery features computed | "
        f"Avg delay: {out['avg_delivery_delay_days'].mean():.1f} days | "
        f"Customers with ≥1 late delivery: "
        f"{(out['delayed_order_count'] > 0).sum():,}"
    )
    return out
