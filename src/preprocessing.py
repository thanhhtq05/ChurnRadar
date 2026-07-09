import pandas as pd

# Các cột categorical cần one-hot encode
CATEGORICAL_COLS = [
    'gender', 'multiple_lines', 'internet_service', 'online_security',
    'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
    'streaming_movies', 'contract', 'payment_method'
]

# Các cột đã là boolean, chỉ cần convert sang int (0/1)
BOOLEAN_COLS = [
    'senior_citizen', 'partner', 'dependents', 'phone_service',
    'paperless_billing', 'churn'
]

NUMERIC_COLS = ['tenure', 'monthly_charges', 'total_charges']

def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # Bỏ customer_id vì không có ý nghĩa dự đoán (chỉ là định danh)
    df = df.drop(columns=['customer_id'])

    # Xử lý missing values ở total_charges (11 dòng thiếu, đã phát hiện lúc EDA)
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())

    # Convert boolean -> int
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Tách X, y
    X = df.drop(columns=['churn'])
    y = df['churn']

    return X, y