# CUSTOMER-LIFETIME-VALUE-CLTV-MODELING
Predict how much each customer will be worth over their lifetime using probabilistic models (BG-NBD + Gamma-Gamma).
[README_project18.md](https://github.com/user-attachments/files/25821673/README_project18.md)
# Customer Lifetime Value (CLTV) Modeling

## Overview
Probabilistic CLTV modeling using BG-NBD (purchase frequency prediction) and Gamma-Gamma (monetary value prediction) models. Predicts how much each customer will be worth over the next 12 months, segments customers by predicted value, and identifies at-risk high-value customers for retention campaigns.

**Built by:** Nithin Kumar Kokkisa — Senior Demand Planner with 12+ years at HPCL managing 180,000 MTPA facility.

---

## Business Problem
Not all customers are equally valuable. Some will generate $5,000 next year, others $50. Marketing budgets should be allocated proportionally to predicted customer value — spend more to retain high-value customers, use automated campaigns for low-value ones. CLTV quantifies each customer's future worth to guide these decisions.

## Two-Model Approach

| Model | Predicts | Input | Output |
|-------|----------|-------|--------|
| **BG-NBD** | How many purchases | frequency, recency, T | Expected purchases in next N days |
| **Gamma-Gamma** | How much per purchase | frequency, monetary_value | Expected average transaction value |
| **CLTV** | Total future value | Both models combined | Dollar value over 12 months |

## CLTV Formula
```
CLTV = Predicted Purchases x Predicted Avg Value x Discount Factor
```

## Key Outputs

- **Probability Alive**: Is the customer still active? (0-100%)
- **Predicted Purchases**: How many purchases in next 12 months
- **CLTV Segments**: Premium, High, Medium, Low based on predicted value
- **At-Risk Customers**: High-value customers with low P(alive) — urgent retention targets

## Visualizations

| Chart | Insight |
|-------|---------|
| Frequency-Recency Matrix | Expected purchases heatmap by customer behavior |
| Probability Alive Matrix | Customer health map — who's still active? |
| CLTV Distribution | How future value is distributed across customers |
| Segment Revenue Share | Premium 25% of customers = 60-70% of future value |

## Install & Run
```bash
pip install lifetimes
# Then run project18_cltv_modeling.py cell by cell in Jupyter
```

## Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib)
- **lifetimes** (BG-NBD, Gamma-Gamma probabilistic models)

---

## About
Part of a **30-project data analytics portfolio**. See [GitHub profile](https://github.com/Kokkisa) for the full portfolio.
