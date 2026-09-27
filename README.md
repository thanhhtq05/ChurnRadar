# Customer Churn Prediction

Dự đoán khả năng khách hàng rời bỏ dịch vụ (churn) cho công ty viễn thông, sử dụng pipeline end-to-end: PostgreSQL → SQL-based EDA → Feature Engineering → Model Comparison → FastAPI Deployment.

## 📌 Problem Statement

Giữ chân khách hàng hiện tại luôn rẻ hơn tìm kiếm khách hàng mới. Project này xây dựng mô hình machine learning để **dự đoán trước khách hàng nào có nguy cơ rời bỏ dịch vụ**, giúp đội chăm sóc khách hàng can thiệp sớm và đúng đối tượng.

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 khách hàng, 21 thuộc tính (demographics, dịch vụ đăng ký, thông tin thanh toán).

## 🏗️ Kiến trúc Project

```
churn-prediction/
├── data/
│   ├── raw/                    # CSV gốc từ Kaggle
│   └── processed/              # Train/test đã xử lý
├── sql/
│   ├── schema.sql               # Định nghĩa bảng customers
│   └── queries/eda_queries.sql  # 6 query khám phá churn pattern
├── src/
│   ├── load_data.py             # Load CSV → PostgreSQL
│   ├── preprocessing.py         # Feature engineering
│   ├── train.py                  # Train Logistic Regression + XGBoost
│   ├── evaluate.py              # Đánh giá model trên tập test
│   ├── predict.py               # Inference logic dùng cho API
│   └── eda_plots.py             # Vẽ biểu đồ EDA
├── notebooks/
│   ├── 01_eda.ipynb             # Insight khám phá dữ liệu
│   └── 02_modeling_experiments.ipynb
├── models/                       # Model đã train (.pkl)
├── api/
│   ├── main.py                  # FastAPI app
│   └── schemas.py               # Pydantic request/response schemas
├── reports/figures/              # Biểu đồ EDA, confusion matrix
├── requirements.txt
└── README.md
```

## 🔧 Tech Stack

- **Database:** PostgreSQL, SQL (window functions, `FILTER`, `CASE WHEN`)
- **Data Processing:** pandas, SQLAlchemy, psycopg2
- **Machine Learning:** scikit-learn, XGBoost
- **API:** FastAPI, Pydantic, Uvicorn
- **Visualization:** matplotlib, seaborn

## 📊 Kết quả chính

### EDA Insights (từ SQL queries)

| Yếu tố | Phát hiện |
|---|---|
| Contract | Month-to-month có churn rate cao gấp ~15 lần Two year |
| Tenure | Khách hàng mới (0-12 tháng) có churn rate 47%, giảm còn 9.5% sau 49+ tháng |
| Payment Method | Electronic check có churn rate cao gấp ~3 lần thanh toán tự động |
| Internet Service | Fiber optic có churn rate 41.9% so với DSL 19% |

### Model Comparison (đánh giá trên tập test, 1,409 mẫu)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | **0.843** | 0.816 |
| F1-score (Churn) | **0.618** | 0.611 |
| Recall (Churn) | **0.79** | 0.71 |
| Precision (Churn) | 0.51 | **0.54** |
| Accuracy | 0.74 | **0.76** |

**→ Chọn Logistic Regression** làm model chính: kết quả tốt hơn trên dữ liệu này, dễ diễn giải hơn, và phù hợp với đặc điểm dataset (quy mô vừa phải, quan hệ giữa feature-churn khá tuyến tính).

### Feature Importance (Top ảnh hưởng churn)

`tenure` (mạnh nhất) → `internet_service_Fiber optic` → `contract_Two year` → `monthly_charges` — khớp nhất quán với insight từ EDA.

Chi tiết đầy đủ về feature selection, xử lý imbalance, và phân tích limitations xem tại `notebooks/02_modeling_experiments.ipynb`.

## 🚀 Cài đặt và Chạy Project

### 1. Yêu cầu
- Python 3.11+
- PostgreSQL đã cài đặt và đang chạy

### 2. Setup môi trường

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

### 3. Setup Database

```bash
psql -U postgres -c "CREATE DATABASE churn_db;"
psql -U postgres -d churn_db -f sql/schema.sql
```

### 4. Load dữ liệu

Tải dataset từ Kaggle, đặt vào `data/raw/`, sau đó:

```bash
python src/load_data.py
```

### 5. Train model

```bash
python src/train.py
```

### 6. Đánh giá model

```bash
python src/evaluate.py
```

### 7. Chạy API

```bash
uvicorn api.main:app --reload
```

Truy cập `http://127.0.0.1:8000/docs` để test endpoint `/predict` qua Swagger UI.

**Ví dụ request:**
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

## 💡 Đề xuất Kinh doanh

1. Tập trung chăm sóc khách hàng trong 12 tháng đầu (giai đoạn rủi ro churn cao nhất)
2. Khuyến khích chuyển đổi sang hợp đồng dài hạn bằng ưu đãi
3. Cải thiện trải nghiệm thanh toán tự động để giảm "ma sát" hàng tháng
4. Rà soát chất lượng dịch vụ Fiber optic
5. Tích hợp model vào CRM để tính điểm rủi ro churn real-time

## ⚠️ Limitations & Next Steps

- XGBoost mới chạy tham số mặc định, chưa tuning (`GridSearchCV`)
- Chưa thử nghiệm Decision Tree/Random Forest
- `total_charges` có đa cộng tuyến với `tenure` × `monthly_charges`, cần thận trọng khi diễn giải
- Dataset là snapshot 1 thời điểm, chưa phản ánh yếu tố mùa vụ
- Hướng mở rộng: SHAP values để giải thích prediction từng khách hàng cụ thể

## 📈 Business Impact

Model đạt Recall 79% cho nhóm churn — nghĩa là bắt được gần 4/5 khách hàng thực sự có nguy cơ rời bỏ, cho phép đội chăm sóc khách hàng can thiệp chủ động thay vì bị động.

---

