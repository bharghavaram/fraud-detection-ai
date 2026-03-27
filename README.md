> **📅 Project Period:** Aug 2024 – Sep 2024 &nbsp;|&nbsp; **Status:** Completed &nbsp;|&nbsp; **Author:** [Bharghava Ram Vemuri](https://github.com/bharghavaram)

# Real-Time Fraud Detection AI

> XGBoost + SHAP explainability + GPT-4o reasoning for financial fraud detection

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)](https://shap.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)

## Overview

A hybrid fraud detection system that combines **classical ML** (XGBoost with SMOTE class balancing) with **LLM narrative reasoning** (GPT-4o) and **SHAP explainability** to provide both accurate predictions and human-understandable explanations.

## Architecture

```
Transaction Input (15 features)
         ↓
XGBoost Model → Fraud Probability (0-1)
         ↓
SHAP Explainer → Top 5 contributing factors
         ↓
GPT-4o Reasoning → Plain-English explanation + red flags
         ↓
Decision: BLOCK | REVIEW | ALLOW
         ↓
Full audit trail with confidence
```

## Feature Engineering (15 dimensions)

- Transaction amount, merchant category, distance from home
- Velocity features: transactions in last 24h, failed attempts
- Account features: card age, account age
- Behavioral: amount deviation from average, velocity score
- Binary: is_international, is_weekend

## Risk Levels

| Probability | Risk Level | Action |
|------------|-----------|--------|
| ≥ 0.80 | CRITICAL | BLOCK immediately |
| ≥ 0.50 | HIGH | REVIEW manually |
| ≥ 0.30 | MEDIUM | REVIEW |
| < 0.30 | LOW | ALLOW |

## Quick Start

```bash
git clone https://github.com/bharghavaram/fraud-detection-ai
cd fraud-detection-ai
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload

# Train the demo model first
curl -X POST "http://localhost:8000/api/v1/fraud/train"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/fraud/train` | Train XGBoost on synthetic data |
| POST | `/api/v1/fraud/predict` | Single transaction prediction |
| POST | `/api/v1/fraud/batch` | Batch predict (max 100) |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/fraud/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5800.00,
    "is_international": true,
    "distance_from_home_km": 1200,
    "failed_attempts_last_hour": 3,
    "card_age_days": 15
  }'
```

## Model Performance (Demo)

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.97 |
| Precision | ~0.94 |
| Recall | ~0.89 |
| F1 Score | ~0.91 |
