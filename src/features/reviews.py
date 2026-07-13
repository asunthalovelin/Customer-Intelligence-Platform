"""
reviews.py — Review Feature Engineering
=========================================
Customer satisfaction signals derived from review scores.
"""

import logging
import pandas as pd

from src.config import POSITIVE_REVIEW_THRESHOLD, NEGATIVE_REVIEW_THRESHOLD

logger = logging.getLogger(__name__)


def compute_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute review-based features.

    Features created:
        avg_review_score      : Mean review score across all orders (1–5 scale)
        positive_review_count : Number of reviews with score >= POSITIVE_REVIEW_THRESHOLD (4)
        negative_review_count : Number of reviews with score <= NEGATIVE_REVIEW_THRESHOLD (2)
    """
    out = df.copy()

    out["avg_review_score"] = out["avg_review_score"].fillna(3.0).round(2)

    # positive_review_count and negative_review_count already computed in joiner
    out["positive_review_count"] = out["positive_review_count"].fillna(0).astype(int)
    out["negative_review_count"] = out["negative_review_count"].fillna(0).astype(int)

    logger.info(
        f"Review features computed | "
        f"Avg score: {out['avg_review_score'].mean():.2f} | "
        f"Customers with negative reviews: "
        f"{(out['negative_review_count'] > 0).sum():,}"
    )
    return out
