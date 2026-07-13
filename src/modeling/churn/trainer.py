"""
trainer.py — Churn Model Trainer
==================================
Loads customer segmented features, defines the churn target (default: >120 days recency),
handles features selection to prevent leakage, prepares pipelines, compares models
(Logistic Regression, Random Forest, XGBoost), and persists the winner.

Design decisions for speed & reliability:
- customer_city is dropped (too many unique values → OHE blows up memory)
- customer_state is kept (27 Brazilian states only → manageable)
- No n_jobs=-1 on RandomizedSearchCV (parallel jobs caused timeout conflicts)
- 3-fold CV with 3 iterations per model to stay within compute budget
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    validate_config,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CHURN_THRESHOLD_DAYS,
    CHURN_MODEL_FILE,
    CHURN_PREPROCESSOR_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core numeric features (no leakage, low cardinality)
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "frequency",
    "monetary_value",
    "total_orders",
    "avg_order_value",
    "max_order_value",
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
]

CATEGORICAL_FEATURES = [
    "preferred_payment_type",  # 5 unique values — safe for OHE
    "customer_state",          # 27 values — safe for OHE
    # customer_city intentionally excluded: ~4,000 unique values → memory explosion
]


def prepare_data() -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Load data, construct churn target, and build the preprocessing pipeline."""
    features_path = PROCESSED_DIR / "customer_features_segmented.parquet"
    logger.info(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)

    # ── Churn target: 1 = churned (inactive > threshold), 0 = active
    y = (df["recency_days"] > CHURN_THRESHOLD_DAYS).astype(int)
    churn_rate = y.mean()
    logger.info(
        f"Churn label created | Threshold: {CHURN_THRESHOLD_DAYS} days | "
        f"Churn rate: {churn_rate:.1%} ({y.sum():,} churned / {len(y):,} total)"
    )

    # ── Build X from explicit feature lists (no leakage, no high-cardinality cats)
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    existing = [c for c in all_features if c in df.columns]
    X = df[existing].copy()

    # ── Impute
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    return X, y, preprocessor


def train_and_compare() -> dict:
    """Train three models, compare on test set, persist the best one."""
    validate_config()
    X, y, preprocessor = prepare_data()

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"Train/Test split: {len(X_train):,} train | {len(X_test):,} test"
    )

    # Fit and save preprocessor
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / CHURN_PREPROCESSOR_FILE)
    logger.info(f"Preprocessor saved → {MODELS_DIR / CHURN_PREPROCESSOR_FILE}")

    # ── Model definitions
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=5, random_state=RANDOM_STATE,
            n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
            n_jobs=1, verbosity=0
        ),
    }

    # ── Hyperparameter grids (small, targeted)
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "Random Forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [5, 10],
        },
        "XGBoost": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
        },
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_roc_auc = 0.0
    best_model_name = None
    best_pipeline = None

    for name, clf in models.items():
        logger.info(f"── Training: {name} ──")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clf),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=3,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,           # single job to avoid multiprocessing overhead
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC AUC":   round(roc_auc_score(y_test, y_prob), 4),
        }

        logger.info(f"  Results: {metrics}")

        results[name] = {
            "metrics": metrics,
            "best_params": {
                k.replace("model__", ""): v
                for k, v in search.best_params_.items()
            },
        }

        if metrics["ROC AUC"] > best_roc_auc:
            best_roc_auc = metrics["ROC AUC"]
            best_model_name = name
            best_pipeline = best_pipe

    # ── Persist best model
    logger.info(
        f"✅ Best model: {best_model_name} | ROC AUC = {best_roc_auc:.4f}"
    )
    joblib.dump(best_pipeline, MODELS_DIR / CHURN_MODEL_FILE)
    logger.info(f"Saved best model → {MODELS_DIR / CHURN_MODEL_FILE}")

    # ── Save metrics JSON
    metrics_path = REPORTS_DIR / "churn_metrics.json"
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
