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

    # Stratify to maintain the correct churn ratio in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline: Logistic Regression with class_weight balanced
    print("\nTraining baseline: Logistic Regression...")
    baseline = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    baseline.fit(X_train_scaled, y_train)

    # Save it to use in the evaluate step
    joblib.dump(baseline, 'models/baseline_logreg.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/feature_columns.pkl')

    
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test_scaled.csv', index=False)

    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() 
    
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train) 

    joblib.dump(xgb_model, 'models/xgboost_churn.pkl')

    # Save the unscaled X_test for XGBoost to use during the evaluate step    X_test.to_csv('data/processed/X_test_unscaled.csv', index=False)

    print("XGBoost model trained and saved to models/xgboost_churn.pkl")
    print("\nBaseline model trained and saved to models/baseline_logreg.pkl")

if __name__ == "__main__":
    main()