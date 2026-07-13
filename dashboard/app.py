"""
app.py — Streamlit Dashboard Application
==========================================
Interactive customer analytics and campaign hub.
Provides business views: Executive Summary, Segmentation, Churn, CLV, and Campaign Center.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import CUSTOMER_INTELLIGENCE_FILE, PROCESSED_DIR

# Set Page Config
st.set_page_config(
    page_title="Customer Intelligence Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode / Harmonious Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #1f77b4;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────
# Data Loading with Cache
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = PROCESSED_DIR / CUSTOMER_INTELLIGENCE_FILE
    if not path.exists():
        st.error(f"Customer intelligence parquet file not found at: {path}")
        st.stop()
    df = pd.read_parquet(path)
    return df

# Main logic
df = load_data()

# Navigation Sidebar
st.sidebar.title("🎯 Intelligence Hub")
page = st.sidebar.radio(
    "Navigate to View:",
    [
        "🏠 Executive Summary",
        "📊 Customer Segmentation",
        "🚨 Churn Prediction",
        "💰 CLV Forecasting",
        "🎯 Recommendation Center"
    ]
)

# ─────────────────────────────────────────────
# Page 1: Executive Summary
# ─────────────────────────────────────────────
if page == "🏠 Executive Summary":
    st.title("🏠 Executive Summary Dashboard")
    st.markdown("Overview of high-level KPIs and business retention exposure.")

    # Calculate overall metrics
    total_customers = len(df)
    total_revenue = df['total_revenue'].sum()
    avg_clv = df['predicted_clv'].mean()
    total_rev_at_risk = df['revenue_at_risk'].sum()
    overall_churn_rate = (df['churn_probability'] > 0.5).mean()

    # Metrics Layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3498db;">
                <div class="metric-label">Total Scored Customers</div>
                <div class="metric-value">{total_customers:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #2ecc71;">
                <div class="metric-label">Total Historical Revenue</div>
                <div class="metric-value">R$ {total_revenue:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #9b59b6;">
                <div class="metric-label">Average Forecasted CLV</div>
                <div class="metric-value">R$ {avg_clv:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e74c3c;">
                <div class="metric-label">Revenue at Risk (Expected)</div>
                <div class="metric-value" style="color: #e74c3c;">R$ {total_rev_at_risk:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Layout for charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue Contribution by Customer Segment")
        segment_rev = df.groupby('customer_segment')['total_revenue'].sum().reset_index()
        fig_pie = px.pie(
            segment_rev, values='total_revenue', names='customer_segment',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Expected Revenue Loss by Churn Level")
        # Bin churn probability into risk buckets
        df['risk_level'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.3, 0.5, 0.8, 1.0],
            labels=['Low Risk (<30%)', 'Moderate Risk (30-50%)', 'High Risk (50-80%)', 'Extreme Risk (>80%)']
        )
        risk_rev = df.groupby('risk_level', observed=False)['revenue_at_risk'].sum().reset_index()
        fig_bar = px.bar(
            risk_rev, x='risk_level', y='revenue_at_risk',
            color='risk_level', color_discrete_map={
                'Low Risk (<30%)': '#2ecc71',
                'Moderate Risk (30-50%)': '#f1c40f',
                'High Risk (50-80%)': '#e67e22',
                'Extreme Risk (>80%)': '#e74c3c'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# Page 2: Customer Segmentation
# ─────────────────────────────────────────────
elif page == "📊 Customer Segmentation":
    st.title("📊 Customer Tribes & Profiles")
    st.markdown("Explore KMeans segments mapped to Olist customer purchasing behaviors.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Segment Volume Share")
        segment_counts = df['customer_segment'].value_counts().reset_index()
        fig_counts = px.pie(
            segment_counts, values='count', names='customer_segment',
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_counts, use_container_width=True)

    with c2:
        st.subheader("2D PCA Customer Projection")
        # Sample down to 3,000 points for quick plotting
        df_sample = df.sample(min(3000, len(df)), random_state=42)
        fig_pca = px.scatter(
            df_sample, x='pca_1', y='pca_2', color='customer_segment',
            opacity=0.6, labels={'pca_1': 'Component 1', 'pca_2': 'Component 2'}
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Tribes Performance Matrix (Averages)")
    segment_metrics = df.groupby('customer_segment').agg(
        customers_count=('customer_unique_id', 'count'),
        avg_recency=('recency_days', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary_value', 'mean'),
        avg_reviews=('avg_review_score', 'mean'),
        avg_delivery_delay=('avg_delivery_delay_days', 'mean')
    ).round(2)
    st.dataframe(segment_metrics, use_container_width=True)

# ─────────────────────────────────────────────
# Page 3: Churn Prediction
# ─────────────────────────────────────────────
elif page == "🚨 Churn Prediction":
    st.title("🚨 Churn Risk Analysis")
    st.markdown("Diagnose churn probability distributions and search accounts for risk assessment.")

    # 1. Churn Risk Distribution Chart
    st.subheader("Churn Probability Risk Distribution")
    fig_hist = px.histogram(
        df, x='churn_probability', nbins=50,
        color_discrete_sequence=['#e74c3c'],
        labels={'churn_probability': 'Predicted Churn Probability'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # 2. Customer Lookup Center
    st.subheader("🔍 Customer Account Profile Lookup")
    lookup_id = st.text_input("Enter Customer Unique ID (e.g. paste from recommendation center):", "").strip()

    if lookup_id:
        cust = df[df['customer_unique_id'] == lookup_id]
        if len(cust) == 0:
            st.error("No customer profile matches that ID. Ensure the complete string is pasted correctly.")
        else:
            cust = cust.iloc[0]
            st.success(f"Customer profile matched successfully!")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Churn Risk Rating", f"{cust['churn_probability']:.1%}")
                st.metric("Expected Revenue Loss", f"R$ {cust['revenue_at_risk']:,.2f}")
            with c2:
                st.metric("Forecasted 6-Month CLV", f"R$ {cust['predicted_clv']:,.2f}")
                st.metric("Customer Segment Label", cust['customer_segment'])
            with c3:
                st.metric("Avg review score rating", f"{cust['avg_review_score']:.1f} ★")
                st.metric("Recommended Action Campaign", cust['recommended_action'])

# ─────────────────────────────────────────────
# Page 4: CLV Forecasting
# ─────────────────────────────────────────────
elif page == "💰 CLV Forecasting":
    st.title("💰 Customer Lifetime Value Forecasts")
    st.markdown("Project future customer value boundaries to filter marketing targets.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Forecasted 6-Month CLV Distribution")
        # Filter out 0 value predictions for cleaner plot
        clv_pos = df[df['predicted_clv'] > 0]
        fig_clv_dist = px.histogram(
            clv_pos, x='predicted_clv', nbins=50,
            color_discrete_sequence=['#16a085'],
            labels={'predicted_clv': 'Predicted Future Spend (R$)'}
        )
        st.plotly_chart(fig_clv_dist, use_container_width=True)

    with c2:
        st.subheader("CLV Distribution across Customer Segments")
        fig_box = px.box(
            df[df['predicted_clv'] > 0], x='customer_segment', y='predicted_clv',
            color='customer_segment', color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────
# Page 5: Recommendation Center
# ─────────────────────────────────────────────
elif page == "🎯 Recommendation Center":
    st.title("🎯 Campaign Recommendation Center")
    st.markdown("Auto-route churn prediction and value classes to target marketing lists.")

    # KPI Summary Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Urgent Retention Targets", len(df[df['recommended_action'] == 'Urgent Retention Campaign']))
    with col2:
        st.metric("Upsell Premium Targets", len(df[df['recommended_action'] == 'Upsell Premium Products']))
    with col3:
        st.metric("Win-Back Targets", len(df[df['recommended_action'] == 'Win-Back Campaign']))

    st.markdown("---")

    # Campaign Targeting Filters
    st.subheader("Filter and Download Marketing Lead Lists")
    
    action_filter = st.selectbox(
        "Select Action Campaign to Filter:",
        df['recommended_action'].unique()
    )
    
    lead_list = df[df['recommended_action'] == action_filter][
        ['customer_unique_id', 'customer_segment', 'churn_probability', 'predicted_clv', 'revenue_at_risk']
    ].sort_values(by='revenue_at_risk', ascending=False)
    
    st.dataframe(lead_list, use_container_width=True)
    
    # Download Button
    csv_data = lead_list.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {action_filter} CSV List",
        data=csv_data,
        file_name=f"lead_list_{action_filter.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
