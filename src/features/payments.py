"""
payments.py — Payment Feature Engineering
==========================================
Payment method preferences and installment usage reveal customer financial
behaviour and purchasing confidence.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute payment-based features.

    Features created:
        avg_installments        : Average maximum installments used across orders
        preferred_payment_type  : Most commonly used payment method (label)
        payment_type_count      : Total count of payment type usages
    """
    out = df.copy()

    out["avg_installments"] = out["avg_installments"].fillna(1.0).round(1)
    out["payment_type_count"] = out["payment_type_count"].fillna(1).astype(int)
    out["preferred_payment_type"] = (
        out["preferred_payment_type"].fillna("unknown").astype(str)
    )

    logger.info(
        f"Payment features computed | "
        f"Most common payment type: "
        f"{out['preferred_payment_type'].value_counts().index[0]}"
    )
    return out
