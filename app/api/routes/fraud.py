"""Fraud Detection AI – API routes."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.services.fraud_service import FraudDetectionService, get_fraud_service

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])

class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    amount: float
    timestamp: Optional[str] = None
    merchant_category_id: Optional[int] = 0
    distance_from_home_km: Optional[float] = 0.0
    transactions_last_24h: Optional[int] = 1
    avg_transaction_amount: Optional[float] = None
    is_international: Optional[bool] = False
    card_age_days: Optional[int] = 365
    account_age_days: Optional[int] = 365
    failed_attempts_last_hour: Optional[int] = 0
    different_merchant_categories_7d: Optional[int] = 3
    velocity_score: Optional[float] = 1.0

class BatchRequest(BaseModel):
    transactions: List[Transaction]

@router.post("/predict")
async def predict(transaction: Transaction, svc: FraudDetectionService = Depends(get_fraud_service)):
    return svc.predict(transaction.model_dump())

@router.post("/batch")
async def batch_predict(req: BatchRequest, svc: FraudDetectionService = Depends(get_fraud_service)):
    if len(req.transactions) > 100:
        raise HTTPException(400, "Max 100 transactions per batch")
    return svc.batch_predict([t.model_dump() for t in req.transactions])

@router.post("/train")
async def train_model(svc: FraudDetectionService = Depends(get_fraud_service)):
    metrics = svc.train_demo_model()
    return {"message": "Model trained successfully", "metrics": metrics}

@router.get("/health")
async def health():
    return {"status": "ok", "service": "Real-Time Fraud Detection AI – XGBoost + LLM Reasoning + SHAP"}
