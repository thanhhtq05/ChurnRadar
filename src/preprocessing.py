import pandas as pd

# Categorical columns need to be one-hot encoded
CATEGORICAL_COLS = [
    'gender', 'multiple_lines', 'internet_service', 'online_security',
    'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
    'streaming_movies', 'contract', 'payment_method'
]

# The columns are already boolean, just need to convert them to int (0/1)
BOOLEAN_COLS = [
    'senior_citizen', 'partner', 'dependents', 'phone_service',
    'paperless_billing', 'churn'
]

NUMERIC_COLS = ['tenure', 'monthly_charges', 'total_charges']

def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # Drop customer_id because it doesn't have predictive value 
    df = df.drop(columns=['customer_id'])

    #Handling missing values in total_charges (11 missing rows, detected during EDA)
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())

    # Convert boolean -> int
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Split X, y
    X = df.drop(columns=['churn'])
    y = df['churn']

    return X, y