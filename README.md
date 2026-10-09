# ChurnRadar: Telecom Customer Churn Prediction

Predicting the likelihood that customers of a telecom company will leave the service (churn), using an end-to-end pipeline: PostgreSQL → SQL-based EDA → Feature Engineering → Model Comparison → FastAPI Deployment → Docker.

## 📌 Problem Statement

Retaining existing customers is always cheaper than acquiring new ones. This project builds a machine learning model to **predict in advance which customers are at risk of leaving the service**, helping the customer care team intervene early and target the right customers.

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 attributes (demographics, subscribed services, billing information).

## 🏗️ Project Structure

```
ChurnRadar/
├── data/
│   ├── raw/                    # Original CSV from Kaggle
│   └── processed/              # Processed train/test sets
├── sql/
│   ├── schema.sql               # Definition of the customers table
│   └── queries/eda_queries.sql  # 6 queries exploring churn patterns
├── src/
│   ├── load_data.py             # Load CSV → PostgreSQL
│   ├── preprocessing.py         # Feature engineering
│   ├── train.py                  # Train Logistic Regression + XGBoost
│   ├── evaluate.py              # Evaluate models on the test set
│   ├── predict.py               # Inference logic used by the API
│   └── eda_plots.py             # EDA plotting
├── notebooks/
│   ├── 01_eda.ipynb             # Data exploration insights
│   └── 02_modeling_experiments.ipynb
├── models/                       # Trained models (.pkl)
├── api/
│   ├── main.py                  # FastAPI app
│   └── schemas.py               # Pydantic request/response schemas
├── reports/figures/              # EDA charts, confusion matrix
├── Dockerfile                    # Container image for the API
├── .dockerignore
├── requirements.txt              # Full environment (EDA + training + API)
├── requirements-api.txt          # Minimal dependencies for the Docker image
└── README.md
```

## 🔧 Tech Stack

- **Database:** PostgreSQL, SQL (window functions, `FILTER`, `CASE WHEN`)
- **Data Processing:** pandas, SQLAlchemy, psycopg2
- **Machine Learning:** scikit-learn, XGBoost
- **API:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker
- **Visualization:** matplotlib, seaborn

## 📊 Key Results

### EDA Insights (from SQL queries)

| Factor | Finding |
|---|---|
| Contract | Month-to-month customers churn at ~15x the rate of Two year customers |
| Tenure | New customers (0-12 months) have a 47% churn rate, dropping to 9.5% after 49+ months |
| Payment Method | Electronic check has a churn rate ~3x higher than automatic payment methods |
| Internet Service | Fiber optic has a 41.9% churn rate vs. 19% for DSL |

### Model Comparison (evaluated on the test set, 1,409 samples)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | **0.843** | 0.816 |
| F1-score (Churn) | **0.618** | 0.611 |
| Recall (Churn) | **0.79** | 0.71 |
| Precision (Churn) | 0.51 | **0.54** |
| Accuracy | 0.74 | **0.76** |

**→ Logistic Regression was chosen** as the main model: it performs better on this data, is easier to interpret, and fits the characteristics of the dataset (moderate size, fairly linear relationship between features and churn).

### Feature Importance (Top drivers of churn)

`tenure` (strongest) → `internet_service_Fiber optic` → `contract_Two year` → `monthly_charges`, consistent with the EDA insights.

For full details on feature selection, imbalance handling, and limitations analysis, see `notebooks/02_modeling_experiments.ipynb`.

## 🚀 Installation and Usage

### 1. Requirements
- Python 3.11+
- PostgreSQL installed and running

### 2. Environment setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

### 3. Database setup

```bash
psql -U postgres -c "CREATE DATABASE churn_db;"
psql -U postgres -d churn_db -f sql/schema.sql
```

### 4. Load data

Download the dataset from Kaggle, place it in `data/raw/`, then run:

```bash
python src/load_data.py
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Evaluate the model

```bash
python src/evaluate.py
```

### 7. Run the API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint through Swagger UI.

**Example request:**
```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 5,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.5,
  "total_charges": 427.5
}
```

**Response:**
```json
{
  "churn_prediction": true,
  "churn_probability": 0.9076,
  "risk_level": "High"
}
```

### 8. Run the API with Docker

The image contains only the API and the trained model files (no PostgreSQL needed to serve predictions).

```
docker build -t churnradar .
docker run -p 8000:8000 churnradar
```

Open `http://127.0.0.1:8000/docs` and call `/predict` with the example request above.

## 💡 Business Recommendations

1. Focus customer care on the first 12 months (the period with the highest churn risk)
2. Encourage switching to long-term contracts through incentives
3. Improve the automatic payment experience to reduce monthly "friction"
4. Review the quality of the Fiber optic service
5. Integrate the model into the CRM to compute churn risk scores in real time

## ⚠️ Limitations & Next Steps

- XGBoost was run with default parameters only and has not been tuned (`GridSearchCV`)
- Decision Tree / Random Forest have not been tested
- `total_charges` is collinear with `tenure` × `monthly_charges`, so interpret it with caution
- The dataset is a single-point-in-time snapshot and does not reflect seasonality
- The API is containerized and runs locally; it is not deployed to a cloud service
- Future extension: SHAP values to explain predictions for individual customers

## 📈 Business Impact

The model achieves 79% Recall for the churn group, meaning it catches nearly 4 out of 5 customers who are truly at risk of leaving, allowing the customer care team to intervene proactively rather than reactively.

---
