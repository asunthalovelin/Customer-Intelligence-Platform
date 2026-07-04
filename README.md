# Executive Customer Intelligence & Retention Report
Generated on: 2026-07-01

This report compiles predictive metrics and customer segments from the Olist Customer Intelligence Platform to answer critical questions regarding customer retention, loyalty, and lifetime value optimization.

---

## Executive Financial Summary

* **Total Scored Customers:** 94,984
* **Aggregate Customer Lifetime Value (Predicted):** R$ 6,529,536.89
* **Total Revenue at Risk (Expected Value):** **R$ 4,294,635.02**
* **Total Churn Risk Exposure (Customers with Churn Prob > 50%):** 84,471 customers (88.9% of base), representing R$ 5,431,483.39 in forecasted future revenue.

---

## Answers to Key Business Questions

### 1. Which customers should be retained?
We prioritize retention targeting by **expected financial loss** using the `revenue_at_risk` metric. Instead of targeting all churned customers, retention budgets should target the top cohort with the highest combined risk and value.

Here are the Top 10 customer retention targets:

| Customer Unique ID | Segment Persona | Predicted CLV | Churn Probability | Expected Revenue at Risk |
|---|---|---|---|---|
| `459bef486812...` | New Customers | R$ 6,664.36 | 78.8% | **R$ 5,252.02** |
| `763c8b1c9c68...` | New Customers | R$ 6,511.51 | 72.6% | **R$ 4,724.19** |
| `f0767ae738c3...` | New Customers | R$ 4,172.01 | 82.5% | **R$ 3,443.26** |
| `3fec1a051777...` | New Customers | R$ 4,018.72 | 74.9% | **R$ 3,009.51** |
| `edde2314c6c3...` | New Customers | R$ 4,578.35 | 61.7% | **R$ 2,825.30** |
| `fa562ef24d41...` | New Customers | R$ 4,135.45 | 67.4% | **R$ 2,789.13** |
| `48e1ac109dec...` | New Customers | R$ 4,560.86 | 55.3% | **R$ 2,524.36** |
| `58483a1c055d...` | New Customers | R$ 4,206.72 | 59.6% | **R$ 2,507.63** |
| `7386ded430d6...` | New Customers | R$ 2,650.29 | 92.6% | **R$ 2,454.54** |
| `129bfc164612...` | New Customers | R$ 3,086.46 | 79.2% | **R$ 2,443.96** |

### 2. Which customers should receive loyalty incentives?
Loyalty incentives should target **high-value, active customers with low churn risk** (churn probability < 30%) in the **VIP** and **Loyal** segments. Offering perks to these stable customers secures their lifetime spend and turns them into brand advocates.
* **Eligible Target Base:** **2,482 customers**
* **Aggregate Value Protected:** **R$ 196,260.37**
* **Strategic Campaigns:** Stored credit, free shipping pass, early access to product launches, and tier upgrades.

### 3. Which customers are highest value?
These represent the absolute top-tier buyers based on predicted CLV. Our database has isolated the following top 10 highest-value customer accounts:

| Customer Unique ID | Segment Persona | Forecasted 6-Month CLV | Churn Probability | Current Status |
|---|---|---|---|---|
| `459bef486812...` | New Customers | **R$ 6,664.36** | 78.8% | 🔴 High Risk |
| `763c8b1c9c68...` | New Customers | **R$ 6,511.51** | 72.6% | 🔴 High Risk |
| `edde2314c6c3...` | New Customers | **R$ 4,578.35** | 61.7% | 🔴 High Risk |
| `48e1ac109dec...` | New Customers | **R$ 4,560.86** | 55.3% | 🟢 Active |
| `a229eba70ec1...` | New Customers | **R$ 4,503.07** | 18.2% | 🟢 Active |
| `ca27f3dac28f...` | New Customers | **R$ 4,312.26** | 13.5% | 🟢 Active |
| `58483a1c055d...` | New Customers | **R$ 4,206.72** | 59.6% | 🟢 Active |
| `f0767ae738c3...` | New Customers | **R$ 4,172.01** | 82.5% | 🔴 High Risk |
| `fa562ef24d41...` | New Customers | **R$ 4,135.45** | 67.4% | 🔴 High Risk |
| `3fec1a051777...` | New Customers | **R$ 4,018.72** | 74.9% | 🔴 High Risk |

### 4. What revenue is at risk from churn?
* **Expected Value Loss:** **R$ 4,294,635.02**
* **Maximum exposure (Active risk):** **R$ 5,431,483.39** across 84,471 customers.
* **Key Risk Concentration:** Over **90%** of the at-risk revenue is concentrated in the single-order segment ("One-Time Buyers" and "New Customers"), representing a critical leak in our activation funnel.

### 5. Which factors most influence churn?
Based on SHAP explainability analysis on our XGBoost classifier, the top factors driving churn are:
1. **`avg_freight_value` (Shipping costs create price friction)** — High freight costs push customers to look for local alternatives.
2. **`avg_delivery_delay_days` (Late packages lead to immediate churn)** — Operational shipping delays are the strongest leading driver of customer attrition.
3. **`avg_review_score` (General transaction rating)** — Directly indicates transaction satisfaction.
4. **`max_order_value` (Higher transaction values hold higher standards)** — Larger transactions demand higher shipping standards.

### 6. Which factors most influence CLV?
Based on SHAP explainability analysis on our Random Forest regressor, the top factors driving CLV are:
1. **`recency_days` (Time elapsed dictates holdout target activation)** — Customers active in the last 6 months dictate holdout spend.
2. **`max_order_value` / `avg_order_value` (Historical basket sizes directly correlate with future spend)** — Past purchase levels are the strongest predictor of future order sizes.
3. **`frequency` (Number of orders placed)** — Higher frequency signals long-term retention.
4. **`customer_tenure_days` (Historical lifespan)** — Established lifespans map to consistent spend.

---

## Strategic Action Plan

1. **Activate the welcome loop:** Target the massive single-order population with onboarding incentives.
2. **Implement dynamic shipping subsidies:** Target customers with high predicted CLV and high freight costs with discounts.
3. **Automate alert systems:** Plug operational metrics (delays, poor reviews) into CRM loops to auto-trigger retention offers.
