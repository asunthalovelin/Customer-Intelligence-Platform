"""
trainer.py — CLV Model Trainer
================================
Constructs a 6-month holdout target (future revenue), trains three regressors
(Linear Regression, Random Forest, XGBoost), compares them, and persists the winner.

Target construction logic:
    cutoff_date = max(last_purchase_date) - 6 months
    Training features : computed from customer_features_segmented (already aggregated lifetime)
    Target (future_revenue): total_revenue for customers whose last purchase is in the
                             6-month holdout window (i.e., they were active near the cutoff)
                             All others get 0 (they did not generate revenue in that window).

Note: Since Olist data is static (no streaming), we approximate holdout revenue as follows:
    - Customers whose last purchase date falls within the final 6 months of the dataset
      are treated as "active" and their total_revenue is used as a proxy for future spend.
    - This is a standard approach for static e-commerce datasets without true train/future splits.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    validate_config,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CLV_MODEL_FILE,
    CLV_PREPROCESSOR_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

# NOTE: monetary_value is intentionally excluded from CLV features.
# monetary_value = total_revenue = the CLV target for holdout customers → data leakage.
# We use behavioural proxies (avg_order_value, recency, frequency) instead.
NUMERIC_FEATURES = [
    "frequency",
    "avg_order_value",       # spend per order — no leakage
    "max_order_value",       # peak spend capability
    "avg_freight_value",
    "customer_tenure_days",
    "avg_days_between_orders",
    "repeat_purchase_rate",
    "avg_review_score",
    "positive_review_count",
    "negative_review_count",
    "avg_installments",
    "payment_type_count",
    "avg_delivery_delay_days",
    "delayed_order_count",
    "on_time_delivery_rate",
    "recency_days",   # valid CLV predictor — not used in target construction
]

CATEGORICAL_FEATURES = [
    "preferred_payment_type",
    "customer_state",
]


def build_clv_target(df: pd.DataFrame) -> pd.Series:
    """
    Construct the CLV target: future revenue in the 6-month holdout window.

    Customers whose last_purchase_date falls within the final 6 months of the dataset
    are assigned their total_revenue as the CLV proxy target.
    All others are assigned 0 (they did not purchase in the holdout period).
    """
    max_date = pd.to_datetime(df["last_purchase_date"]).max()
    cutoff = max_date - relativedelta(months=6)
    logger.info(f"Dataset max date: {max_date.date()} | Holdout cutoff: {cutoff.date()}")

    last_purchase = pd.to_datetime(df["last_purchase_date"])
    in_holdout = last_purchase >= cutoff

    clv_target = pd.Series(0.0, index=df.index)
    clv_target[in_holdout] = df.loc[in_holdout, "total_revenue"].values

    logger.info(
        f"Holdout customers (active in last 6 months): {in_holdout.sum():,} "
        f"({in_holdout.mean():.1%}) | Avg CLV target: R${clv_target[in_holdout].mean():.2f}"
    )
    return clv_target


def prepare_data() -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    features_path = PROCESSED_DIR / "customer_features_segmented.parquet"
    logger.info(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)

    y = build_clv_target(df)

    # Drop non-feature columns
    drop_cols = [
        "customer_unique_id", "cluster", "customer_segment",
        "first_purchase_date", "last_purchase_date",
        "avg_delivery_date", "avg_estimated_delivery",
        "pca_1", "pca_2", "delayed_orders",
        "total_revenue",       # target leakage
        "total_item_price",    # redundant with total_revenue
        "total_freight",       # redundant
    ]

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    existing = [c for c in all_features if c in df.columns]
    X = df[existing].copy()

    # Impute
    for c in NUMERIC_FEATURES:
        if c in X.columns:
            X[c] = X[c].fillna(0)
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].fillna("unknown").astype(str)

    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    logger.info(
        f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} cols "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical)"
    )

    from sklearn.preprocessing import OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    return X, y, preprocessor


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2":   round(float(r2), 4),
    }


def train_and_compare() -> dict:
    validate_config()
    X, y, preprocessor = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train/Test split: {len(X_train):,} train | {len(X_test):,} test")

    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / CLV_PREPROCESSOR_FILE)
    logger.info(f"Preprocessor saved → {MODELS_DIR / CLV_PREPROCESSOR_FILE}")

    models = {
        "Linear Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0
        ),
    }

    param_grids = {
        "Linear Regression": {"model__alpha": [0.1, 1.0, 10.0]},
        "Random Forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [5, 10],
        },
        "XGBoost": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
        },
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_rmse = float("inf")
    best_model_name = None
    best_pipeline = None

    for name, reg in models.items():
        logger.info(f"── Training: {name} ──")
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", reg)])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=3,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        y_pred = best_pipe.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)   # CLV cannot be negative

        metrics = evaluate_regression(y_test.values, y_pred)
        logger.info(f"  Results: {metrics}")

        results[name] = {
            "metrics": metrics,
            "best_params": {
                k.replace("model__", ""): v
                for k, v in search.best_params_.items()
            },
        }

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model_name = name
            best_pipeline = best_pipe

    logger.info(f"✅ Best model: {best_model_name} | RMSE = {best_rmse:.4f}")
    joblib.dump(best_pipeline, MODELS_DIR / CLV_MODEL_FILE)
    logger.info(f"Saved best model → {MODELS_DIR / CLV_MODEL_FILE}")

    metrics_path = REPORTS_DIR / "clv_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved metrics → {metrics_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    train_and_compare()
