# load_data.py
import pandas as pd
from sqlalchemy import create_engine
import os

# connect database
DB_USER = "postgres"
DB_PASSWORD = "" 
DB_HOST = "localhost"
DB_PORT = ""
DB_NAME = "churn_db"

CSV_PATH = "data/raw/data_customer_churn.csv"  

COLUMN_MAPPING = {
    'customerID': 'customer_id',
    'gender': 'gender',
    'SeniorCitizen': 'senior_citizen',
    'Partner': 'partner',
    'Dependents': 'dependents',
    'tenure': 'tenure',
    'PhoneService': 'phone_service',
    'MultipleLines': 'multiple_lines',
    'InternetService': 'internet_service',
    'OnlineSecurity': 'online_security',
    'OnlineBackup': 'online_backup',
    'DeviceProtection': 'device_protection',
    'TechSupport': 'tech_support',
    'StreamingTV': 'streaming_tv',
    'StreamingMovies': 'streaming_movies',
    'Contract': 'contract',
    'PaperlessBilling': 'paperless_billing',
    'PaymentMethod': 'payment_method',
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges': 'total_charges',
    'Churn': 'churn',
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAPPING)

    # Handle TotalCharges
    df['total_charges'] = df['total_charges'].replace(' ', pd.NA)
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    # Normalize Yes/No -> boolean
    yes_no_cols = ['partner', 'dependents', 'phone_service',
                   'paperless_billing', 'churn']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': True, 'No': False})

    df['senior_citizen'] = df['senior_citizen'].astype(bool)

    return df

def main():
    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    print(f"Raw shape: {df.shape}")
    df = clean_data(df)

    # Check for missing values after cleaning
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0])

    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    print("Loading into PostgreSQL...")
    df.to_sql(
        'customers',
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=500
    )

    print(f"Done. Loaded {len(df)} rows into 'customers' table.")

if __name__ == "__main__":
    main()