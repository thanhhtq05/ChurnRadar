import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from .schemas import CustomerInput, ChurnPrediction  # thêm dấu . phía trước
from predict import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API dự đoán khả năng khách hàng rời bỏ dịch vụ telecom",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Visit /docs for API documentation."}

@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerInput):
    try:
        data = customer.model_dump()
        result = predict_churn(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")