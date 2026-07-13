"""
joiner.py — Table Join Logic
==============================
Joins the 5 Olist tables into a single customer-level analytical base dataset.
Every join step logs row counts for full traceability.
"""

import logging

import pandas as pd

from src.config import VALID_ORDER_STATUSES, PROCESSED_DIR, CUSTOMER_BASE_FILE

logger = logging.getLogger(__name__)


def _log_shape(label: str, df: pd.DataFrame) -> None:
    logger.info(f"  [{label}] → {df.shape[0]:,} rows × {df.shape[1]} columns")


def filter_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep only orders with actionable statuses."""
    before = len(orders)
    df = orders[orders["order_status"].isin(VALID_ORDER_STATUSES)].copy()
    after = len(df)
    dropped = before - after
    logger.info(
        f"Order filter: {before:,} → {after:,} rows "
        f"(dropped {dropped:,} cancelled/unavailable)"
    )
    return df


def aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order_items to one row per order_id."""
    agg = (
        order_items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            total_item_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            max_item_price=("price", "max"),
        )
        .reset_index()
    )
    logger.info(f"Order items aggregated: {agg.shape[0]:,} unique orders")
    return agg


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payments to one row per order_id."""
    # Mode payment type per order
    payment_mode = (
        payments.groupby("order_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .reset_index()
        .rename(columns={"payment_type": "payment_type_mode"})
    )
    # Numeric aggregations
    payment_agg = (
        payments.groupby("order_id")
        .agg(
            total_payment_value=("payment_value", "sum"),
            max_installments=("payment_installments", "max"),
            payment_type_count=("payment_type", "nunique"),
        )
        .reset_index()
    )
    agg = payment_agg.merge(payment_mode, on="order_id", how="left")
    logger.info(f"Payments aggregated: {agg.shape[0]:,} unique orders")
    return agg


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews to one row per order_id (take the latest review if multiple)."""
    agg = (
        reviews.sort_values("review_creation_date")
        .groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count"),
        )
        .reset_index()
    )
    logger.info(f"Reviews aggregated: {agg.shape[0]:,} unique orders")
    return agg


def build_order_level(
    orders: pd.DataFrame,
    order_items_agg: pd.DataFrame,
    payments_agg: pd.DataFrame,
    reviews_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all order-level aggregates into a single order-level dataset."""
    logger.info("Building order-level dataset ...")

    df = orders.copy()
    _log_shape("orders (filtered)", df)

    df = df.merge(order_items_agg, on="order_id", how="left")
    _log_shape("+ order_items", df)

    df = df.merge(payments_agg, on="order_id", how="left")
    _log_shape("+ payments", df)

    df = df.merge(reviews_agg, on="order_id", how="left")
    _log_shape("+ reviews", df)

    return df


def collapse_to_customer_level(
    customers: pd.DataFrame,
    order_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge customers with order-level data, then collapse to one row
    per customer_unique_id.
    """
    logger.info("Collapsing to customer level ...")

    # Map customer_id → customer_unique_id
    df = order_level.merge(
        customers[["customer_id", "customer_unique_id",
                   "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
    )
    _log_shape("+ customers (unique id mapped)", df)

    # Aggregate to customer_unique_id
    customer_df = (
        df.groupby("customer_unique_id")
        .agg(
            # Order counts
            total_orders=("order_id", "nunique"),
            # Revenue
            total_revenue=("total_payment_value", "sum"),
            total_item_price=("total_item_price", "sum"),
            total_freight=("total_freight_value", "sum"),
            max_order_value=("total_payment_value", "max"),
            avg_order_value=("total_payment_value", "mean"),
            avg_freight_value=("total_freight_value", "mean"),
            # Timestamps
            first_purchase_date=("order_purchase_timestamp", "min"),
            last_purchase_date=("order_purchase_timestamp", "max"),
            # Delivery
            avg_delivery_date=("order_delivered_customer_date", "mean"),
            avg_estimated_delivery=("order_estimated_delivery_date", "mean"),
            # Reviews
            avg_review_score=("review_score", "mean"),
            # Payments
            avg_installments=("max_installments", "mean"),
            payment_type_count=("payment_type_count", "sum"),
            # Payment type: most common across all orders
            preferred_payment_type=("payment_type_mode",
                                    lambda x: x.mode().iloc[0]
                                    if not x.mode().empty else "unknown"),
            # Location
            customer_state=("customer_state", "first"),
            customer_city=("customer_city", "first"),
            # Delivery delay (computed at order level below)
            delayed_orders=("order_id", "count"),  # placeholder, recalculated
        )
        .reset_index()
    )

    # ── Delivery delay per order → aggregate at customer level
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    delay_agg = (
        df.groupby("customer_unique_id")
        .agg(
            avg_delivery_delay_days=("delivery_delay_days", "mean"),
            delayed_order_count=("delivery_delay_days",
                                 lambda x: (x > 0).sum()),
            total_deliveries=("delivery_delay_days", "count"),
        )
        .reset_index()
    )

    customer_df = customer_df.merge(delay_agg, on="customer_unique_id", how="left")
    customer_df["on_time_delivery_rate"] = (
        1 - customer_df["delayed_order_count"] / customer_df["total_deliveries"]
    ).clip(0, 1)

    # ── Review score buckets (computed at order level)
    df["is_positive_review"] = df["review_score"] >= 4
    df["is_negative_review"] = df["review_score"] <= 2

    review_agg = (
        df.groupby("customer_unique_id")
        .agg(
            positive_review_count=("is_positive_review", "sum"),
            negative_review_count=("is_negative_review", "sum"),
        )
        .reset_index()
    )
    customer_df = customer_df.merge(review_agg, on="customer_unique_id", how="left")

    _log_shape("customer-level base", customer_df)
    return customer_df


def build_customer_base(raw: dict) -> pd.DataFrame:
    """
    Full pipeline: filter → aggregate → join → collapse.
    Returns the customer-level base dataset.
    """
    logger.info("=" * 60)
    logger.info("Building Customer Base Dataset")
    logger.info("=" * 60)

    orders_filtered = filter_orders(raw["orders"])
    items_agg = aggregate_order_items(raw["order_items"])
    payments_agg = aggregate_payments(raw["payments"])
    reviews_agg = aggregate_reviews(raw["reviews"])

    order_level = build_order_level(
        orders_filtered, items_agg, payments_agg, reviews_agg
    )

    customer_base = collapse_to_customer_level(raw["customers"], order_level)

    # Save
    out_path = PROCESSED_DIR / CUSTOMER_BASE_FILE
    customer_base.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved customer base → {out_path}")
    logger.info(f"   Final shape: {customer_base.shape[0]:,} customers × "
                f"{customer_base.shape[1]} columns")

    return customer_base
