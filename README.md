# Customer Intelligence & Revenue Retention Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-intelligence-platform-kc9mnguxvwxushucn6infv.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20Random%20Forest-red.svg)](https://xgboost.readthedocs.io/)

An end-to-end customer intelligence system built on 94k+ Olist transactions that clusters buyers, predicts churn risk (0.82 AUC), forecasts future lifetime value (CLV), explains individual risk factors using SHAP, and generates automated retention lead campaign targets.

---

## 🚀 Key Accomplishments (Summary)

* **Built an end-to-end Customer Intelligence Platform** that scores **94k+ accounts**, yielding **R$ 6.5M** in forecasted customer lifetime value (CLV) and isolating **R$ 4.2M** in expected revenue-at-risk.
* **Developed an XGBoost churn classifier achieving a 0.8225 ROC-AUC** (with **95.8% Recall** to maximize retention lead capture), outperforming baseline Logistic Regression (0.5864).
* **Constructed a 6-month holdout timeline CLV forecast** utilizing a Random Forest Regressor ($R^2$ of 0.9975, MAE of R$ 0.61), dynamically defining customer value tiers.
* **Implemented a SHAP model explainability engine** to capture global drivers (freight cost and delivery delay as top churn metrics) and generate individual customer waterfall plots for customer service prioritization.
* **Authored a multi-page interactive Streamlit dashboard** containing Executive summaries, PCA segmentation visuals, predictive distribution checks, customer lookups, and lead-filtering download centers to guide marketing routing.

---

## 📊 Key Insights & Visualizations

### 1. Customer Tribes (KMeans Clustering)
We mapped our customer base into 5 distinct behavioral persona segments using scaled RFM metrics. High-value "VIP" and "New" cohorts represent over 30% of total database value.
*(Once you upload the figures folder, this will display on your profile)*
![Customer Segments](reports/figures/segment_profiles_bar.png)

### 2. Explainability & Churn Drivers (SHAP)
Using SHAP values, we open the "black box" of the XGBoost classifier, showing that logistics performance (freight cost and delivery delay) are the leading triggers of attrition.
![SHAP Churn Beeswarm](reports/figures/shap_churn_beeswarm.png)

---

## 📁 Repository Overview
- `src/` - Production-style pipeline modules (loaders, joiners, trainers).
- `dashboard/` - User interface dashboard application.
- `notebooks/` - Step-by-step presentation notebooks (`01` to `06`).
- [Detailed Business Report](reports/business_report.md) - Deep dive analysis answering strategic retention questions.
