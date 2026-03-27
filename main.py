"""Fraud Detection AI – FastAPI Entry Point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.fraud import router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

app = FastAPI(
    title="Real-Time Fraud Detection AI",
    description="Hybrid fraud detection combining XGBoost ML model with GPT-4o narrative reasoning and SHAP explainability. Classifies transactions as BLOCK/REVIEW/ALLOW with full explanations.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "Real-Time Fraud Detection AI",
        "version": "1.0.0",
        "model_stack": {"ml_model": "XGBoost + SMOTE", "explainability": "SHAP", "reasoning": "GPT-4o"},
        "risk_levels": ["LOW → ALLOW", "MEDIUM → REVIEW", "HIGH → REVIEW", "CRITICAL → BLOCK"],
        "features": ["15-dimensional feature engineering", "SMOTE class balancing", "SHAP top-5 factor explanation", "LLM narrative reasoning", "Real-time single + batch prediction"],
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
