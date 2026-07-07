# 📊 Customer Behaviour Analysis & Churn Prediction System

A full-stack Streamlit application for customer analytics, segmentation, and churn prediction — built for internship/final-year project demonstrations.

## Overview

This system lets a business upload (or auto-generate) customer transaction data, automatically cleans and engineers features (RFM + CLV), segments customers with K-Means clustering, trains and compares churn-prediction models, and produces business recommendations and downloadable reports — all through an interactive dashboard.

## Features

- **Role-based login** (Admin / User) with separate permissions
- **Dashboard** with KPI cards: customers, revenue, churn rate, CLV, AOV, and more
- **Dataset management**: generate synthetic data, upload your own CSVs, preview & clean
- **Automatic preprocessing**: missing values, duplicates, outlier capping, RFM + CLV feature engineering
- **15+ interactive EDA charts** (Plotly) covering demographics, sales trends, categories, channels, correlations
- **Customer segmentation** via K-Means with Elbow method, silhouette score, PCA visualization, and named segments (Champions, Loyal, At Risk, Lost, etc.)
- **Churn prediction**: Logistic Regression, Decision Tree, Random Forest — trained on transactional RFM/CLV features plus independent behavioural signals (support tickets, engagement score, satisfaction rating) so churn is a genuinely learned pattern rather than a restated feature. Auto-compared and best model selected (saved via Joblib)
- **Persistence**: processed data and trained models are saved to `data/` and `models/` and automatically reloaded on the next app restart
- **Automatic Customer Archetype Management**: every customer is automatically classified as `new`, `regular`, `loyal`, `at_risk`, or `high_value` based on signup recency and purchase history — no manual assignment. Adding a new customer or transaction (via the **Add New Customer** / **Add New Transaction** admin pages) instantly recalculates that customer's archetype and refreshes the dashboard, segmentation, and churn model
- **Individual customer lookup**: search, view profile/history, live churn probability & risk level
- **Business recommendation engine** mapped to segment and risk
- **Reports**: Customer, Segment, Churn, Sales, Revenue & Insights reports — exportable as CSV, Excel, or PDF

## Technologies Used

- **Frontend:** Streamlit, custom CSS
- **Backend:** Python
- **Libraries:** Pandas, NumPy, Plotly, Matplotlib, Seaborn, Scikit-learn, Joblib, XlsxWriter, ReportLab
- **ML:** K-Means Clustering, Logistic Regression, Decision Tree, Random Forest

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Demo Credentials
| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |
| User  | user     | user123   |

## Screenshots

_Add screenshots here after running the app locally (Dashboard, Segmentation, Churn Prediction pages recommended)._

## Project Structure

```
churn_app/
├── app.py                 # Main Streamlit application
├── requirements.txt
├── README.md
├── data/                   # Persisted processed datasets (auto-saved on first run)
├── models/                 # Saved ML models (joblib, auto-reloaded on restart)
├── outputs/                 # Exported processed data
├── reports/                 # Generated report files
├── figures/                 # Saved chart images
├── assets/                  # Static assets
└── src/
    ├── generate_data.py    # Synthetic data generator
    ├── preprocessing.py    # Cleaning + RFM/CLV feature engineering
    ├── segmentation.py     # K-Means clustering & cluster naming
    ├── churn_model.py      # Model training, comparison, prediction
    ├── recommendations.py  # Business recommendation engine
    ├── reports.py           # CSV / Excel / PDF report builders (with embedded charts)
    ├── storage.py            # Persistence layer (save/load pipeline state & models)
    ├── archetype.py           # Automatic customer archetype classification rules
    └── customer_management.py # Add customer / add transaction (canonical CSV CRUD)
```

## Future Enhancements

- Persistent database (PostgreSQL) instead of in-memory/session state
- Hashed password storage + multi-admin user management
- Deep-learning churn models (e.g., XGBoost, LightGBM)
- Real-time data ingestion via API
- Scheduled automated report emailing
- A/B testing module for retention campaigns

## Author

Generated as a demonstration project for internship / academic submission.
