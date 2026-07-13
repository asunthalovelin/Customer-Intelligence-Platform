"""
loader.py — Raw CSV Loaders
============================
Loads each Olist CSV with consistent dtypes, logging, and null reporting.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CUSTOMERS_FILE,
    ORDERS_FILE,
    ORDER_ITEMS_FILE,
    PAYMENTS_FILE,
    REVIEWS_FILE,
    PRODUCTS_FILE,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


def _load_csv(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Generic CSV loader with logging and null reporting."""
    path: Path = RAW_DATA_DIR / filename
    logger.info(f"Loading {filename} ...")
    df = pd.read_csv(path, parse_dates=parse_dates)
    logger.info(f"  → {df.shape[0]:,} rows × {df.shape[1]} columns loaded")

    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        logger.warning(f"  ⚠ Nulls detected in {filename}:")
        for col, cnt in null_cols.items():
            pct = cnt / len(df) * 100
            logger.warning(f"      {col}: {cnt:,} ({pct:.1f}%)")
    else:
        logger.info(f"  ✓ No nulls detected in {filename}")
    return df


def load_customers() -> pd.DataFrame:
    """Load olist_customers_dataset.csv."""
    return _load_csv(CUSTOMERS_FILE)


def load_orders() -> pd.DataFrame:
    """Load olist_orders_dataset.csv with all timestamp columns parsed."""
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    return _load_csv(ORDERS_FILE, parse_dates=date_cols)


def load_order_items() -> pd.DataFrame:
    """Load olist_order_items_dataset.csv."""
    return _load_csv(ORDER_ITEMS_FILE, parse_dates=["shipping_limit_date"])


def load_payments() -> pd.DataFrame:
    """Load olist_order_payments_dataset.csv."""
    return _load_csv(PAYMENTS_FILE)


def load_reviews() -> pd.DataFrame:
    """Load olist_order_reviews_dataset.csv."""
    date_cols = ["review_creation_date", "review_answer_timestamp"]
    return _load_csv(REVIEWS_FILE, parse_dates=date_cols)


def load_products() -> pd.DataFrame:
    """Load olist_products_dataset.csv."""
    return _load_csv(PRODUCTS_FILE)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all required raw tables and return as a dictionary."""
    logger.info("=" * 60)
    logger.info("Loading all raw Olist datasets")
    logger.info("=" * 60)
    return {
        "customers": load_customers(),
        "orders": load_orders(),
        "order_items": load_order_items(),
        "payments": load_payments(),
        "reviews": load_reviews(),
    }
