import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)
import matplotlib.pyplot as plt

def load_test_data():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')
    return X_test, X_test_scaled, y_test

def evaluate_model(model, X_test, y_test, model_name, use_proba_threshold=0.5):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print('='*50)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    filename = model_name.lower().replace(' ', '_')
    plt.savefig(f'reports/figures/confusion_matrix_{filename}.png', dpi=150)
    plt.close()

    return {'model': model_name, 'auc': auc, 'f1': f1}

def main():
    X_test, X_test_scaled, y_test = load_test_data()
    X_test_unscaled = pd.read_csv('data/processed/X_test_unscaled.csv')

    baseline = joblib.load('models/baseline_logreg.pkl')
    xgb_model = joblib.load('models/xgboost_churn.pkl')

    results = []
    results.append(evaluate_model(baseline, X_test_scaled, y_test, 'Logistic Regression'))
    results.append(evaluate_model(xgb_model, X_test_unscaled, y_test, 'XGBoost'))

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    for r in results:
        print(f"{r['model']}: AUC={r['auc']:.4f}, F1={r['f1']:.4f}")
        
        
if __name__ == "__main__":
    main()
    
import pandas as pd
import joblib

model = joblib.load('models/baseline_logreg.pkl')
feature_names = joblib.load('models/feature_columns.pkl')

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df.head(10))