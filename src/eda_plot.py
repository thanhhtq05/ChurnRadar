import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Cấu hình ---
DB_USER = "postgres"
DB_PASSWORD = "051207"  # thay password thật của em
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churn_db"

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    df = pd.read_sql("SELECT * FROM customers", engine)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def plot_churn_by_contract(df):
    plt.figure(figsize=(6, 4))
    churn_by_contract = df.groupby('contract')['churn'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, palette='Reds_r')
    plt.ylabel('Churn Rate (%)')
    plt.title('Churn Rate by Contract Type')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/churn_by_contract.png', dpi=150)
    plt.close()
    print("Saved: churn_by_contract.png")

def plot_tenure_distribution(df):
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x='tenure', hue='churn', bins=30, kde=True, element='step')
    plt.title('Tenure Distribution by Churn Status')
    plt.xlabel('Tenure (months)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tenure_distribution.png', dpi=150)
    plt.close()
    print("Saved: tenure_distribution.png")

def plot_churn_by_payment(df):
    plt.figure(figsize=(7, 4))
    churn_by_payment = df.groupby('payment_method')['churn'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=churn_by_payment.values, y=churn_by_payment.index, palette='Reds_r')
    plt.xlabel('Churn Rate (%)')
    plt.title('Churn Rate by Payment Method')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/churn_by_payment.png', dpi=150)
    plt.close()
    print("Saved: churn_by_payment.png")

def plot_monthly_charges_boxplot(df):
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x='churn', y='monthly_charges', palette='Set2')
    plt.title('Monthly Charges: Churn vs Retained')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/monthly_charges_boxplot.png', dpi=150)
    plt.close()
    print("Saved: monthly_charges_boxplot.png")

def main():
    df = load_data()
    plot_churn_by_contract(df)
    plot_tenure_distribution(df)
    plot_churn_by_payment(df)
    plot_monthly_charges_boxplot(df)
    print("\nAll plots saved to reports/figures/")

if __name__ == "__main__":
    main()