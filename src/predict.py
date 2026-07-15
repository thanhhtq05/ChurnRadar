import pandas as pd
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'baseline_logreg.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

CATEGORICAL_COLS = [
    'gender', 'multiple_lines', 'internet_service', 'online_security',
    'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
    'streaming_movies', 'contract', 'payment_method'
]

def preprocess_single_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Convert Yes/No -> 1/0 cho các cột boolean (giống lúc train)
    yes_no_cols = ['partner', 'dependents', 'phone_service', 'paperless_billing']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode giống lúc train
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Đảm bảo đủ cột đúng như lúc train (cột nào thiếu -> điền 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Sắp xếp đúng thứ tự cột như lúc train
    df = df[feature_columns]

    return df

def predict_churn(data: dict) -> dict:
    X = preprocess_single_input(data)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0][1]
    prediction = bool(model.predict(X_scaled)[0])

    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(proba), 4),
        "risk_level": risk
    }