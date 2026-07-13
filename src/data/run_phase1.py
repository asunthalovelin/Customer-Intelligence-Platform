"""
run_phase1.py — Run Phase 1 Data Engineering Pipeline
======================================================
Executes data loading, cleaning, joining, collapsing, data quality auditing,
and documentation of assumptions. Saves data_quality_report_phase1.md.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path to ensure src import works
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

from src.config import validate_config, PROCESSED_DIR, REPORTS_DIR, CUSTOMER_BASE_FILE
from src.data.loader import load_all_raw
from src.data.joiner import build_customer_base

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("run_phase1")

def run_pipeline():
    validate_config()
    
    # 1. Load raw data
    raw = load_all_raw()
    
    # Track statistics for validation
    stats = {}
    stats['raw_customers_count'] = len(raw['customers'])
    stats['raw_customers_unique_count'] = raw['customers']['customer_unique_id'].nunique()
    stats['raw_orders_count'] = len(raw['orders'])
    stats['raw_order_items_count'] = len(raw['order_items'])
    stats['raw_payments_count'] = len(raw['payments'])
    stats['raw_reviews_count'] = len(raw['reviews'])
    
    logger.info("Raw table counts:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:,}")
        
    # 2. Execute joins and aggregation to customer base
    customer_base = build_customer_base(raw)
    
    # 3. Compile Data Quality Metrics
    stats['final_customers_count'] = len(customer_base)
    stats['final_customers_unique_count'] = customer_base['customer_unique_id'].nunique()
    stats['final_columns_count'] = len(customer_base.columns)
    
    # Check null percentages in the customer base
    null_counts = customer_base.isnull().sum()
    null_pct = (null_counts / len(customer_base) * 100).round(2)
    null_report = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage (%)': null_pct
    }).sort_values(by='Null Percentage (%)', ascending=False)
    
    # 4. Generate the Data Quality Report
    report_path = REPORTS_DIR / "data_quality_report_phase1.md"
    logger.info(f"Writing Data Quality Report to {report_path}...")
    
    # Formulate report markdown
    report_md = f"""# Phase 1 Data Quality Report & Join Validation
Generated on: 2026-06-04

This report validates the creation of the customer-level analytical dataset for the Customer Intelligence Platform.

## Data Pipeline Flow Summary

| Table | Raw Row Count | Description |
|---|---|---|
| **Customers** | {stats['raw_customers_count']:,} | Base table mapping customer_id to customer_unique_id. |
| **Orders** | {stats['raw_orders_count']:,} | Order transaction metadata (timestamps, status). |
| **Order Items** | {stats['raw_order_items_count']:,} | Line-item prices and freight values. |
| **Payments** | {stats['raw_payments_count']:,} | Payment transaction values and installments. |
| **Reviews** | {stats['raw_reviews_count']:,} | Customer satisfaction scores. |
| **Customer Base (Final)** | {stats['final_customers_count']:,} | One row per unique customer, aggregated. |

## Join and Aggregation Validation

* **Raw Unique Customers:** {stats['raw_customers_unique_count']:,}
* **Final Customer Rows:** {stats['final_customers_count']:,}
* **Retention/Match Rate:** {stats['final_customers_count'] / stats['raw_customers_unique_count'] * 100:.2f}%
  * *Note: The slight difference is due to filtering out cancelled or unavailable orders (retaining only delivered, shipped, processing, invoiced).*
* **Uniqueness check:** Number of duplicate `customer_unique_id` rows in final base = **{customer_base.duplicated(subset=['customer_unique_id']).sum()}** (Must be 0).

## Analytical Dataset Features Schema

Below is the null-rate audit of the collapsed customer analytical dataset:

| Feature Name | Null Count | Null % | Data Quality Status |
|---|---|---|---|
"""
    for col, row in null_report.iterrows():
        status = "✅ OK" if row['Null Percentage (%)'] < 5.0 else "⚠️ Check needed"
        report_md += f"| `{col}` | {int(row['Null Count']):,} | {row['Null Percentage (%)']:.2f}% | {status} |\n"
        
    report_md += f"""
## Key Assumptions and Business Logic Decisions

1. **Active Order Definition:** Only orders with status `delivered`, `shipped`, `processing`, or `invoiced` are kept. Cancelled or unavailable orders are discarded to avoid skewing historical revenue metrics.
2. **Order-Level Consolidation:**
   - Order Items: Prices and freight are aggregated via `sum()`.
   - Payments: Value is summed; installments are represented by `max()`. The preferred payment method is resolved by taking the mode payment type.
   - Reviews: The average review score is taken across all order items/reviews for that order.
3. **Reference Point for Recency:** The dataset's maximum observed `order_purchase_timestamp` ({customer_base['last_purchase_date'].max()}) is used as the hypothetical "today" reference point to calculate days since last purchase, avoiding timeline leakage.
4. **Handling of Missing Delivery Timestamps:** Orders with missing delivery dates (e.g. processing/shipped status) result in `NaT` values for delivery time and delivery delay, which are handled dynamically using fill values (median delay or 0 days) during feature engineering.

## Verification Action Item
- Customer Base file size on disk: **{os.path.getsize(PROCESSED_DIR / CUSTOMER_BASE_FILE) / (1024*1024):.2f} MB**
- Saved as: `processed/{CUSTOMER_BASE_FILE}`

"""
    with open(report_path, "w") as f:
        f.write(report_md)
        
    logger.info("Phase 1 Data Engineering successfully completed!")
    print("✅ Phase 1 complete. Base dataset written and data quality report saved.")

if __name__ == "__main__":
    run_pipeline()
