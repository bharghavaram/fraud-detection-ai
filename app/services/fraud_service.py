"""
Real-Time Fraud Detection AI – XGBoost + LLM reasoning + SHAP explainability.
Combines classical ML with LLM narrative explanations for financial fraud detection.
"""
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

FRAUD_REASONING_PROMPT = """You are a financial fraud analyst. Explain this transaction's fraud risk assessment.

TRANSACTION DETAILS:
{transaction}

ML MODEL PREDICTION:
- Fraud Probability: {fraud_prob:.1%}
- Risk Level: {risk_level}
- Decision: {decision}

TOP RISK FACTORS (SHAP):
{shap_factors}

Provide:
1. Plain-English explanation of why this transaction is/isn't suspicious
2. Specific red flags identified
3. Recommended action (BLOCK | REVIEW | ALLOW)
4. Confidence in recommendation (HIGH | MEDIUM | LOW)

Format as JSON: {{"explanation": "...", "red_flags": [...], "recommendation": "BLOCK|REVIEW|ALLOW", "confidence": "HIGH|MEDIUM|LOW", "analyst_notes": "..."}}"""

FEATURE_NAMES = [
    "amount", "hour_of_day", "day_of_week", "merchant_category",
    "distance_from_home_km", "transactions_last_24h", "avg_transaction_amount",
    "amount_deviation_from_avg", "is_international", "is_weekend",
    "card_age_days", "account_age_days", "failed_attempts_last_hour",
    "different_merchant_categories_7d", "velocity_score",
]


class FraudDetectionService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = None
        self.scaler = None
        self._init_or_load_model()

    def _init_or_load_model(self):
        model_path = Path(settings.MODEL_PATH)
        if model_path.exists():
            import joblib
            data = joblib.load(model_path)
            self.model = data["model"]
            self.scaler = data.get("scaler")
            logger.info("Loaded model from %s", model_path)
        else:
            logger.info("No saved model found, will train on first request or use heuristics")

    def train_demo_model(self) -> dict:
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        from imblearn.over_sampling import SMOTE
        import xgboost as xgb
        import joblib

        # Generate synthetic fraud data
        np.random.seed(42)
        n_samples = 10000
        n_fraud = int(n_samples * 0.02)  # 2% fraud rate

        legit = pd.DataFrame({
            "amount": np.random.lognormal(4, 1, n_samples - n_fraud),
            "hour_of_day": np.random.randint(6, 23, n_samples - n_fraud),
            "day_of_week": np.random.randint(0, 7, n_samples - n_fraud),
            "merchant_category": np.random.randint(0, 20, n_samples - n_fraud),
            "distance_from_home_km": np.random.exponential(10, n_samples - n_fraud),
            "transactions_last_24h": np.random.poisson(3, n_samples - n_fraud),
            "avg_transaction_amount": np.random.lognormal(4, 0.5, n_samples - n_fraud),
            "amount_deviation_from_avg": np.random.normal(0, 50, n_samples - n_fraud),
            "is_international": np.random.binomial(1, 0.05, n_samples - n_fraud),
            "is_weekend": np.random.binomial(1, 0.3, n_samples - n_fraud),
            "card_age_days": np.random.randint(30, 3000, n_samples - n_fraud),
            "account_age_days": np.random.randint(90, 5000, n_samples - n_fraud),
            "failed_attempts_last_hour": np.random.poisson(0.1, n_samples - n_fraud),
            "different_merchant_categories_7d": np.random.randint(1, 8, n_samples - n_fraud),
            "velocity_score": np.random.exponential(1, n_samples - n_fraud),
            "label": 0,
        })

        fraud = pd.DataFrame({
            "amount": np.random.lognormal(6, 1.5, n_fraud),
            "hour_of_day": np.random.choice([0, 1, 2, 3, 23], n_fraud),
            "day_of_week": np.random.randint(0, 7, n_fraud),
            "merchant_category": np.random.randint(0, 20, n_fraud),
            "distance_from_home_km": np.random.exponential(200, n_fraud),
            "transactions_last_24h": np.random.poisson(15, n_fraud),
            "avg_transaction_amount": np.random.lognormal(4, 0.5, n_fraud),
            "amount_deviation_from_avg": np.random.normal(500, 200, n_fraud),
            "is_international": np.random.binomial(1, 0.6, n_fraud),
            "is_weekend": np.random.binomial(1, 0.3, n_fraud),
            "card_age_days": np.random.randint(1, 60, n_fraud),
            "account_age_days": np.random.randint(1, 90, n_fraud),
            "failed_attempts_last_hour": np.random.poisson(3, n_fraud),
            "different_merchant_categories_7d": np.random.randint(5, 20, n_fraud),
            "velocity_score": np.random.exponential(8, n_fraud),
            "label": 1,
        })

        df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
        X = df[FEATURE_NAMES].values
        y = df["label"].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        self.model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= settings.FRAUD_THRESHOLD).astype(int)

        metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_pred_proba), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "training_samples": len(X_train),
        }

        Path(settings.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({"model": self.model, "scaler": self.scaler}, settings.MODEL_PATH)
        logger.info("Model trained and saved. Metrics: %s", metrics)
        return metrics

    def _extract_features(self, transaction: dict) -> np.ndarray:
        now = datetime.fromisoformat(transaction.get("timestamp", datetime.utcnow().isoformat()))
        amount = float(transaction.get("amount", 0))
        avg_amount = float(transaction.get("avg_transaction_amount", amount))
        features = [
            amount,
            now.hour,
            now.weekday(),
            transaction.get("merchant_category_id", 0),
            float(transaction.get("distance_from_home_km", 0)),
            int(transaction.get("transactions_last_24h", 1)),
            avg_amount,
            amount - avg_amount,
            int(transaction.get("is_international", 0)),
            int(now.weekday() >= 5),
            int(transaction.get("card_age_days", 365)),
            int(transaction.get("account_age_days", 365)),
            int(transaction.get("failed_attempts_last_hour", 0)),
            int(transaction.get("different_merchant_categories_7d", 3)),
            float(transaction.get("velocity_score", 1.0)),
        ]
        return np.array(features, dtype=float).reshape(1, -1)

    def _get_shap_factors(self, features: np.ndarray) -> List[dict]:
        if not settings.SHAP_ENABLED or self.model is None:
            return []
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(features)
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            factors = []
            for i, (name, val) in enumerate(zip(FEATURE_NAMES, values[0])):
                factors.append({"feature": name, "value": round(float(features[0][i]), 3), "shap_impact": round(float(val), 4)})
            return sorted(factors, key=lambda x: abs(x["shap_impact"]), reverse=True)[:5]
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)
            return []

    def _heuristic_score(self, transaction: dict) -> float:
        score = 0.0
        amount = float(transaction.get("amount", 0))
        if amount > settings.MAX_AMOUNT_THRESHOLD:
            score += 0.3
        if transaction.get("is_international"):
            score += 0.2
        if transaction.get("failed_attempts_last_hour", 0) > 2:
            score += 0.25
        if transaction.get("transactions_last_24h", 0) > 10:
            score += 0.15
        if transaction.get("distance_from_home_km", 0) > 500:
            score += 0.1
        return min(score, 0.99)

    def predict(self, transaction: dict) -> dict:
        features = self._extract_features(transaction)

        # Get fraud probability
        if self.model and self.scaler:
            features_scaled = self.scaler.transform(features)
            fraud_prob = float(self.model.predict_proba(features_scaled)[0][1])
        else:
            fraud_prob = self._heuristic_score(transaction)

        # Risk classification
        if fraud_prob >= settings.HIGH_RISK_THRESHOLD:
            risk_level = "CRITICAL"
            decision = "BLOCK"
        elif fraud_prob >= settings.FRAUD_THRESHOLD:
            risk_level = "HIGH"
            decision = "REVIEW"
        elif fraud_prob >= 0.3:
            risk_level = "MEDIUM"
            decision = "REVIEW"
        else:
            risk_level = "LOW"
            decision = "ALLOW"

        # SHAP explanation
        shap_factors = self._get_shap_factors(features)

        # LLM reasoning
        llm_analysis = {}
        if settings.LLM_REASONING_ENABLED:
            try:
                shap_text = "\n".join([f"  - {f['feature']}: {f['value']} (impact: {f['shap_impact']:+.4f})" for f in shap_factors])
                resp = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": FRAUD_REASONING_PROMPT.format(
                        transaction=json.dumps(transaction, indent=2),
                        fraud_prob=fraud_prob,
                        risk_level=risk_level,
                        decision=decision,
                        shap_factors=shap_text or "Not available",
                    )}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                llm_analysis = json.loads(resp.choices[0].message.content)
            except Exception as exc:
                logger.error("LLM reasoning failed: %s", exc)

        return {
            "transaction_id": transaction.get("transaction_id", "unknown"),
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk_level,
            "decision": llm_analysis.get("recommendation", decision),
            "confidence": llm_analysis.get("confidence", "MEDIUM"),
            "explanation": llm_analysis.get("explanation", ""),
            "red_flags": llm_analysis.get("red_flags", []),
            "analyst_notes": llm_analysis.get("analyst_notes", ""),
            "top_risk_factors": shap_factors,
            "model_used": "xgboost" if self.model else "heuristic",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def batch_predict(self, transactions: List[dict]) -> dict:
        results = [self.predict(t) for t in transactions]
        blocked = sum(1 for r in results if r["decision"] == "BLOCK")
        reviewed = sum(1 for r in results if r["decision"] == "REVIEW")
        return {
            "total": len(results),
            "blocked": blocked,
            "flagged_for_review": reviewed,
            "allowed": len(results) - blocked - reviewed,
            "fraud_rate": round((blocked + reviewed) / max(len(results), 1) * 100, 1),
            "results": results,
        }


_service: Optional[FraudDetectionService] = None
def get_fraud_service() -> FraudDetectionService:
    global _service
    if _service is None:
        _service = FraudDetectionService()
    return _service
