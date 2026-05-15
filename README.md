> **📅 Period:** Aug 2024 – Sep 2024 &nbsp;|&nbsp; **Author:** [Bharghava Ram Vemuri](https://github.com/bharghavaram)

<div align="center">

# 🛡️ Fraud Detection AI

### Real-Time Financial Fraud Detection · XGBoost + SHAP + GPT-4o Reasoning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![CI](https://github.com/bharghavaram/fraud-detection-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/bharghavaram/fraud-detection-ai/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat)](https://xgboost.readthedocs.io)

</div>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/bharghavaram/fraud-detection-ai/main/docs/images/demo.svg" alt="fraud-detection-ai demo" width="820"/>
</div>

--- 🎯 Problem Statement

Financial fraud costs $485B annually. Rule-based systems miss novel fraud patterns; ML models give opaque decisions that compliance teams cannot explain to regulators; and real-time latency requirements (<100ms) make deep learning impractical. This system combines XGBoost for sub-50ms prediction, SHAP for feature-level explainability, and GPT-4o for human-readable narrative reasoning — delivering BLOCK/REVIEW/ALLOW decisions with full audit trails.

---

## 🏗️ Architecture

```
Transaction Input (15 features)
        │
   ┌────▼────────────────────────────┐
   │  Feature Engineering            │
   │  velocity · time · merchant     │
   └────┬────────────────────────────┘
        │
   ┌────▼──────┐      ┌──────────────┐
   │ XGBoost   │      │  SHAP Values │
   │ Classifier│─────►│  Explainer   │
   └────┬──────┘      └──────┬───────┘
        │                    │
   ┌────▼────────────────────▼──────┐
   │  GPT-4o Narrative Reasoner     │
   │  "This transaction was flagged │
   │   because of high velocity..." │
   └────────────────────────────────┘
        │
   Decision: BLOCK | REVIEW | ALLOW + Audit Trail
```

---

## 📁 Project Structure

```
fraud-detection-ai/
├── main.py
├── app/
│   ├── services/
│   │   ├── detection_service.py   # XGBoost inference pipeline
│   │   ├── shap_service.py        # SHAP explainability
│   │   ├── reasoning_service.py   # GPT-4o narrative generation
│   │   └── features_service.py    # 15-feature engineering
│   └── api/routes/
│       ├── detect.py
│       └── explain.py
├── tests/
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/bharghavaram/fraud-detection-ai.git
cd fraud-detection-ai
pip install -r requirements.txt
cp .env.example .env   # Add OPENAI_API_KEY (optional, for narratives)
uvicorn main:app --reload
```

---

## 🤖 Model & Algorithm Details

| Component | Algorithm | Dataset | Details |
|-----------|-----------|---------|---------|
| Fraud Classifier | XGBoost (gradient boosting) | IEEE-CIS Fraud Detection (590K transactions) | 15 engineered features, 500 trees, max_depth=6 |
| Class Imbalance | SMOTE oversampling | — | Synthetic minority oversampling to 50:50 ratio |
| Explainability | SHAP TreeExplainer | — | Per-prediction feature contribution scores |
| Narrative | GPT-4o | — | Top 3 SHAP features → human-readable explanation |
| Decision Threshold | Optimised F1 | — | BLOCK >0.85, REVIEW 0.4–0.85, ALLOW <0.4 |

**15 Engineered Features:** transaction_amount, hour_of_day, day_of_week, merchant_category, distance_from_home, transactions_last_1h, transactions_last_24h, avg_amount_30d, amount_vs_avg_ratio, new_merchant_flag, cross_border_flag, card_present, device_fingerprint_match, billing_shipping_match, account_age_days

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Real-time fraud detection with decision |
| POST | `/detect/batch` | Batch detection (up to 100 transactions) |
| POST | `/explain` | SHAP + GPT-4o narrative for a transaction |
| GET | `/model/metrics` | Current model performance metrics |

---

## 💡 Sample Input → Output

**Request:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"amount":4999.99,"merchant_category":"gift_cards","hour":3,"transactions_last_1h":7,"new_merchant":true}'
```
**Response:**
```json
{
  "decision": "BLOCK",
  "fraud_probability": 0.94,
  "confidence": "HIGH",
  "top_risk_factors": [
    {"feature":"transactions_last_1h","shap_value":0.38,"description":"7 transactions in 1 hour (3× normal)"},
    {"feature":"merchant_category","shap_value":0.29,"description":"Gift cards are high-risk category"},
    {"feature":"hour","shap_value":0.18,"description":"3 AM — unusual transaction time"}
  ],
  "narrative": "This transaction was blocked due to extremely high velocity (7 transactions in 1 hour), suspicious merchant category (gift cards), and unusual timing (3 AM). Combined fraud probability: 94%.",
  "latency_ms": 43
}
```

---

## 📊 Evaluation Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.974 |
| Precision (fraud class) | 0.91 |
| Recall (fraud class) | 0.88 |
| F1 Score | 0.895 |
| False Positive Rate | 4.2% |
| Inference latency | <50ms |
| SHAP explanation time | <15ms |

---

## ⚙️ Environment Variables

```env
OPENAI_API_KEY=sk-...
FRAUD_THRESHOLD=0.85
REVIEW_THRESHOLD=0.40
MAX_BATCH_SIZE=100
```

---

## 🧪 Testing · 🗺️ Roadmap · 📄 License

```bash
pytest tests/ -v
```
**Roadmap:** Real-time Kafka stream integration · Graph neural network for relationship fraud · A/B testing framework for threshold tuning · Regulatory reporting export

MIT License — see [LICENSE](LICENSE). Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
