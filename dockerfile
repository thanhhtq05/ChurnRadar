FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs (training/EDA libraries are not required to serve predictions)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# API code, inference logic and the trained artifacts
COPY api/ api/
COPY src/predict.py src/predict.py
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
