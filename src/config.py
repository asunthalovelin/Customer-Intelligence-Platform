"""
config.py — Central Configuration for Customer Intelligence Platform
======================================================================

All project-wide constants, thresholds, and directory paths are defined here.
No values should be hardcoded anywhere else in the codebase.

Churn Threshold Explanation
----------------------------
A customer is considered **churned** if they have not made a purchase
within CHURN_THRESHOLD_DAYS after their most recent order.

Supported values: 60, 90, 120, 180 days
Default: 120 days

    - 60 days  → Aggressive definition; flags customers who haven't bought in 2 months.
                  Suited for high-frequency categories (e.g., groceries, cosmetics).
    - 90 days  → Moderate definition; common in e-commerce literature.
    - 120 days → Conservative default; balances precision and recall for general retail.
    - 180 days → Lenient definition; suited for low-frequency categories (e.g., electronics).

To change the churn window, modify only CHURN_THRESHOLD_DAYS below.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# DATA DIRECTORIES
# ─────────────────────────────────────────────
RAW_DATA_DIR: Path = PROJECT_ROOT / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR: Path = PROJECT_ROOT / "dashboard"

# ─────────────────────────────────────────────
# RAW DATA FILE NAMES
# ─────────────────────────────────────────────
CUSTOMERS_FILE: str = "olist_customers_dataset.csv"
ORDERS_FILE: str = "olist_orders_dataset.csv"
ORDER_ITEMS_FILE: str = "olist_order_items_dataset.csv"
PAYMENTS_FILE: str = "olist_order_payments_dataset.csv"
REVIEWS_FILE: str = "olist_order_reviews_dataset.csv"
PRODUCTS_FILE: str = "olist_products_dataset.csv"

# ─────────────────────────────────────────────
# PROCESSED DATA FILE NAMES
# ─────────────────────────────────────────────
CUSTOMER_BASE_FILE: str = "customer_base.parquet"
CUSTOMER_FEATURES_FILE: str = "customer_features.parquet"
CUSTOMER_SEGMENTED_FILE: str = "customer_features_segmented.parquet"
CUSTOMER_INTELLIGENCE_FILE: str = "customer_intelligence_final.parquet"

# ─────────────────────────────────────────────
# MODEL FILE NAMES
# ─────────────────────────────────────────────
CHURN_MODEL_FILE: str = "churn_model.joblib"
CHURN_PREPROCESSOR_FILE: str = "churn_preprocessor.joblib"
CLV_MODEL_FILE: str = "clv_model.joblib"
CLV_PREPROCESSOR_FILE: str = "clv_preprocessor.joblib"
SEGMENTATION_MODEL_FILE: str = "kmeans_segmenter.joblib"
SEGMENTATION_SCALER_FILE: str = "rfm_scaler.joblib"

# ─────────────────────────────────────────────
# CHURN CONFIGURATION
# ─────────────────────────────────────────────
# ✏️  TO CHANGE THE CHURN WINDOW: modify only this value.
# Supported: 60, 90, 120, 180
CHURN_THRESHOLD_DAYS: int = 120

# Valid options (for documentation and validation)
VALID_CHURN_THRESHOLDS: list[int] = [60, 90, 120, 180]

# ─────────────────────────────────────────────
# CLV CONFIGURATION
# ─────────────────────────────────────────────
# Holdout window: last N months of data used as CLV target period
CLV_HOLDOUT_MONTHS: int = 6

# ─────────────────────────────────────────────
# SEGMENTATION CONFIGURATION
# ─────────────────────────────────────────────
KMEANS_MIN_K: int = 2
KMEANS_MAX_K: int = 10
# Final k will be chosen automatically via elbow + silhouette analysis
KMEANS_RANDOM_STATE: int = 42

# ─────────────────────────────────────────────
# MODEL TRAINING CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20          # 80/20 train-test split
CV_FOLDS: int = 5                # StratifiedKFold cross-validation
HYPERPARAM_ITERATIONS: int = 30  # RandomizedSearchCV iterations

# ─────────────────────────────────────────────
# REVIEW SCORE THRESHOLDS
# ─────────────────────────────────────────────
POSITIVE_REVIEW_THRESHOLD: int = 4   # Score >= 4 → positive
NEGATIVE_REVIEW_THRESHOLD: int = 2   # Score <= 2 → negative

# ─────────────────────────────────────────────
# CLV TIER PERCENTILE THRESHOLDS
# ─────────────────────────────────────────────
CLV_LOW_PERCENTILE: float = 33.0
CLV_HIGH_PERCENTILE: float = 67.0

# ─────────────────────────────────────────────
# CHURN PROBABILITY THRESHOLDS (for recommendations)
# ─────────────────────────────────────────────
CHURN_HIGH_RISK_THRESHOLD: float = 0.60
CHURN_IMMEDIATE_THRESHOLD: float = 0.80
CHURN_MODERATE_THRESHOLD: float = 0.50

# ─────────────────────────────────────────────
# ORDER STATUS FILTER
# ─────────────────────────────────────────────
# Only keep orders in these statuses for analysis
VALID_ORDER_STATUSES: list[str] = [
    "delivered",
    "shipped",
    "processing",
    "invoiced",
]

# ─────────────────────────────────────────────
# STREAMLIT DASHBOARD
# ─────────────────────────────────────────────
STREAMLIT_PORT: int = 8501

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def validate_config() -> None:
    """Validate that all required directories exist and config values are sane."""
    assert CHURN_THRESHOLD_DAYS in VALID_CHURN_THRESHOLDS, (
        f"CHURN_THRESHOLD_DAYS must be one of {VALID_CHURN_THRESHOLDS}, "
        f"got {CHURN_THRESHOLD_DAYS}"
    )
    assert 0 < TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
    assert CV_FOLDS >= 2, "CV_FOLDS must be at least 2"
    # Ensure output directories exist
    for directory in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    validate_config()
    print("✅ Configuration validated successfully.")
    print(f"   Project root   : {PROJECT_ROOT}")
    print(f"   Churn threshold: {CHURN_THRESHOLD_DAYS} days")
    print(f"   CLV holdout    : {CLV_HOLDOUT_MONTHS} months")
    print(f"   Random state   : {RANDOM_STATE}")
