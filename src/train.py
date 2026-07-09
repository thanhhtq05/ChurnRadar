import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from xgboost import XGBClassifier

from preprocessing import prepare_features

DB_USER = "postgres"
DB_PASSWORD = "051207"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churn_db"

os.makedirs("models", exist_ok=True)

def load_data():
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return pd.read_sql("SELECT * FROM customers", engine)

def main():
    print("Loading data...")
    df = load_data()
    X, y = prepare_features(df)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")

    # Stratify để giữ đúng tỷ lệ churn ở cả train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Scale numeric features (Logistic Regression nhạy với scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline: Logistic Regression với class_weight balanced
    print("\nTraining baseline: Logistic Regression...")
    baseline = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    baseline.fit(X_train_scaled, y_train)

    # Lưu lại để dùng ở bước evaluate
    joblib.dump(baseline, 'models/baseline_logreg.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/feature_columns.pkl')

    # Lưu train/test split để evaluate.py dùng lại (đảm bảo evaluate đúng trên test set)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test_scaled.csv', index=False)

    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # tương đương class_weight='balanced'
    
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)  # dùng X_train gốc, KHÔNG dùng bản đã scale

    joblib.dump(xgb_model, 'models/xgboost_churn.pkl')

    # Lưu X_test bản KHÔNG scale để XGBoost dùng ở bước evaluate
    X_test.to_csv('data/processed/X_test_unscaled.csv', index=False)

    print("XGBoost model trained and saved to models/xgboost_churn.pkl")
    print("\nBaseline model trained and saved to models/baseline_logreg.pkl")

if __name__ == "__main__":
    main()