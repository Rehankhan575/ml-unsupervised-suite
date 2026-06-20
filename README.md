# Customer Segmentation — Unsupervised Learning Suite

A production-grade customer segmentation system built on UK e-commerce transaction data. Transforms raw transactional history into actionable customer segments using RFM feature engineering, K-Means clustering, and a FastAPI inference layer containerized with Docker.

## Business Problem

An e-commerce retailer needs to understand its customer base to design targeted marketing strategies. Instead of treating all 4,338 customers identically, we identify four distinct behavioral segments and prescribe specific actions for each.

## Results

| Segment | Customers | Revenue Share | Strategy |
|---------|-----------|---------------|----------|
| Champions | 951 (21.9%) | 74.7% | Reward and retain — VIP program, no discounts |
| Loyal | 1321 (30.4%) | 13.3% | Upsell — bundles and tiered loyalty points |
| At Risk | 1058 (24.4%) | 9.9% | Reactivation — win-back email with limited discount |
| Hibernating | 1008 (23.2%) | 2.0% | Minimal spend — one final offer then suppress |

**Key insight:** 21.9% of customers generate 74.7% of revenue. Retaining Champions is the single highest-ROI action available.

**Revenue opportunity:** Reactivating 10% of At Risk customers at their historical average spend generates an estimated £88,553 in additional revenue.

## Tech Stack

Python · Pandas · Scikit-learn · Scipy · Matplotlib · Seaborn · MLflow · FastAPI · Docker

## Project Structure

```
customer-segmentation/
├── notebook/
│   └── unsupervised.ipynb       # EDA, RFM engineering, clustering, profiling
├── app/
│   ├── main.py                  # FastAPI routes
│   ├── model.py                 # Preprocessing and prediction logic
│   └── schemas.py               # Pydantic request/response models
├── models/                      # Saved scaler, KMeans, PCA artifacts
├── mlflow/                      # MLflow SQLite experiment database
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Approach

**1. EDA** — Analysed 541,909 transactions across 4,338 customers. Removed cancelled orders, guest transactions, and invalid prices. Final dataset: 397,884 clean transactions.

**2. RFM Feature Engineering** — Computed Recency, Frequency, Monetary, AvgOrderValue, and CustomerLifetime per customer. All features heavily right-skewed (Monetary skew: 19.3).

**3. Preprocessing** — Winsorized at 99th percentile → log1p transform → RobustScaler. Monetary skew reduced from 19.3 to 0.4.

**4. Clustering** — Compared K-Means, Hierarchical (Ward), and DBSCAN. K-Means with K=4 selected based on elbow method, silhouette analysis, and business interpretability. Hierarchical ARI of 0.57 validates cluster structure.

**5. Dimensionality Reduction** — PCA captures 87.1% variance in 2 components. t-SNE confirms clean segment separation.

**6. MLflow Tracking** — All K values (2-6) logged with parameters and metrics for reproducible comparison.

## Algorithm Comparison

| Algorithm | Silhouette | Clusters | Verdict |
|-----------|------------|----------|---------|
| K-Means | 0.2794 | 4 | Primary model |
| Hierarchical | 0.2632 | 4 | Validates K-Means (ARI 0.57) |
| DBSCAN | 0.2917 | 3 | Not suitable — flags Champions as noise |

## Quick Start

**Run with Docker:**
```bash
docker build -t customer-segmentation-api .
docker run -p 8000:8000 customer-segmentation-api
```

**API docs:** http://127.0.0.1:8000/docs

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"recency": 15, "frequency": 12, "monetary": 8500, "avg_order_value": 708, "customer_lifetime": 280}'
```

**Run notebook locally:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/unsupervised.ipynb
```

**MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000 --host 127.0.0.1
```

## Dataset

UK-based online retail transactions 2010-2011. Source: [UCI Machine Learning Repository via Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data). Download `data.csv` and place in `data/` folder.
