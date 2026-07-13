"""
kmeans_segmenter.py — Customer Segmentation Module
===================================================
Applies KMeans clustering on scaled RFM features, performs PCA for visualization,
profiles segments, and maps them to business-centric persona labels.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import (
    MODELS_DIR,
    PROCESSED_DIR,
    CUSTOMER_FEATURES_FILE,
    CUSTOMER_SEGMENTED_FILE,
    SEGMENTATION_MODEL_FILE,
    SEGMENTATION_SCALER_FILE,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)

def perform_segmentation(k: int = 5) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Perform KMeans segmentation on RFM features.
    
    Args:
        k: Number of clusters (default: 5)
        
    Returns:
        DataFrame with cluster labels, fitted KMeans model, and StandardScaler.
    """
    logger.info("Starting customer segmentation pipeline...")
    
    # 1. Load customer features
    features_path = PROCESSED_DIR / CUSTOMER_FEATURES_FILE
    df = pd.read_parquet(features_path)
    
    # 2. Extract and scale RFM features
    rfm_cols = ['recency_days', 'frequency', 'monetary_value']
    X = df[rfm_cols].copy()
    
    # Log-transform to handle skewness before scaling
    X_log = np.log1p(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # 3. Fit KMeans model
    logger.info(f"Fitting KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Save models
    joblib.dump(kmeans, MODELS_DIR / SEGMENTATION_MODEL_FILE)
    joblib.dump(scaler, MODELS_DIR / SEGMENTATION_SCALER_FILE)
    logger.info(f"Saved models to {MODELS_DIR}")
    
    # 5. Compute PCA for 2D visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    # 6. Map cluster IDs to descriptive business personas
    df = map_cluster_labels(df, kmeans, scaler)
    
    # Save segmented features dataset
    out_path = PROCESSED_DIR / CUSTOMER_SEGMENTED_FILE
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved segmented customer dataset to {out_path}")
    
    return df, kmeans, scaler

def map_cluster_labels(df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler) -> pd.DataFrame:
    """
    Dynamically map numeric cluster IDs to business personas based on their centroid profiles.
    
    Persona rules based on cluster center rankings:
    - VIP Customers: Lowest recency, highest frequency, highest monetary.
    - Loyal Customers: Low/moderate recency, high frequency, moderate/high monetary.
    - New Customers: Low recency, low frequency, low monetary.
    - At Risk Customers: High recency, moderate frequency, moderate monetary.
    - One-Time Buyers: High recency, low frequency, low monetary.
    """
    # Calculate raw centroids (inverse scale the log centroids)
    centroids_log = kmeans.cluster_centers_
    # Inverse transform standard scaling
    centroids_log_orig = scaler.inverse_transform(centroids_log)
    # Inverse transform log1p (exp(x) - 1)
    centroids = np.expm1(centroids_log_orig)
    
    centroid_df = pd.DataFrame(
        centroids, 
        columns=['recency_days', 'frequency', 'monetary_value']
    )
    centroid_df['cluster'] = range(len(centroids))
    
    # Sort and rank clusters to label them logically
    # Rank lowest recency as best (1), highest freq as best (1), highest monetary as best (1)
    rec_ranks = centroid_df['recency_days'].rank(ascending=True) # lower is better (more recent)
    freq_ranks = centroid_df['frequency'].rank(ascending=False)  # higher is better
    mon_ranks = centroid_df['monetary_value'].rank(ascending=False) # higher is better
    
    # Assign names based on profile properties
    assigned_labels = {}
    
    # VIP: Best frequency and monetary, low recency
    vip_idx = np.argmin(centroids[:, 0] + (100 / centroids[:, 1]) + (1000 / centroids[:, 2]))
    assigned_labels[vip_idx] = "VIP Customers"
    
    # Find one-time buyers (high recency, frequency close to 1)
    # Filter out VIP
    remaining_indices = [i for i in range(len(centroids)) if i != vip_idx]
    
    # One-Time Buyers (highest recency, lowest frequency)
    otb_idx = remaining_indices[np.argmax(centroids[remaining_indices, 0])]
    assigned_labels[otb_idx] = "One-Time Buyers"
    
    remaining_indices = [i for i in remaining_indices if i != otb_idx]
    
    # Loyal Customers (high frequency, low recency)
    loyal_idx = remaining_indices[np.argmax(centroids[remaining_indices, 1])]
    assigned_labels[loyal_idx] = "Loyal Customers"
    
    remaining_indices = [i for i in remaining_indices if i != loyal_idx]
    
    # New Customers (lowest recency, low frequency)
    new_idx = remaining_indices[np.argmin(centroids[remaining_indices, 0])]
    assigned_labels[new_idx] = "New Customers"
    
    remaining_indices = [i for i in remaining_indices if i != new_idx]
    
    # Last remaining must be "At Risk Customers" or "Bargain Customers"
    if len(remaining_indices) > 0:
        at_risk_idx = remaining_indices[0]
        assigned_labels[at_risk_idx] = "At Risk Customers"
        
    # Apply labels mapping
    df['customer_segment'] = df['cluster'].map(assigned_labels)
    
    # Print profile summary to console
    logger.info("Cluster Business Profiles (Centroids):")
    for cid, label in sorted(assigned_labels.items()):
        rec = centroid_df.loc[cid, 'recency_days']
        freq = centroid_df.loc[cid, 'frequency']
        mon = centroid_df.loc[cid, 'monetary_value']
        logger.info(f"  Cluster {cid} ({label}): Recency={rec:.1f} days, Freq={freq:.2f}, Monetary=R${mon:.2f}")
        
    return df
