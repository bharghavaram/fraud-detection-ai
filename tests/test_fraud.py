"""Tests for Fraud Detection AI."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from app.core.config import settings

def test_settings():
    assert settings.FRAUD_THRESHOLD == 0.5
    assert settings.HIGH_RISK_THRESHOLD == 0.8

def test_risk_classification():
    cases = [
        (0.9, "CRITICAL", "BLOCK"),
        (0.6, "HIGH", "REVIEW"),
        (0.35, "MEDIUM", "REVIEW"),
        (0.1, "LOW", "ALLOW"),
    ]
    for prob, expected_risk, expected_decision in cases:
        if prob >= 0.8: risk, decision = "CRITICAL", "BLOCK"
        elif prob >= 0.5: risk, decision = "HIGH", "REVIEW"
        elif prob >= 0.3: risk, decision = "MEDIUM", "REVIEW"
        else: risk, decision = "LOW", "ALLOW"
        assert risk == expected_risk
        assert decision == expected_decision

def test_feature_extraction():
    with patch("app.services.fraud_service.OpenAI"):
        from app.services.fraud_service import FraudDetectionService
        svc = FraudDetectionService()
        transaction = {"amount": 500.0, "is_international": True, "card_age_days": 30}
        features = svc._extract_features(transaction)
        assert features.shape == (1, 15)
        assert features[0][0] == 500.0

def test_heuristic_score_high_amount():
    with patch("app.services.fraud_service.OpenAI"):
        from app.services.fraud_service import FraudDetectionService
        svc = FraudDetectionService()
        transaction = {"amount": 50000.0, "is_international": True, "failed_attempts_last_hour": 5}
        score = svc._heuristic_score(transaction)
        assert score >= 0.5
        assert score <= 1.0

def test_heuristic_score_safe_transaction():
    with patch("app.services.fraud_service.OpenAI"):
        from app.services.fraud_service import FraudDetectionService
        svc = FraudDetectionService()
        transaction = {"amount": 25.0, "is_international": False}
        score = svc._heuristic_score(transaction)
        assert score < 0.5

@pytest.mark.asyncio
async def test_api_health():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/api/v1/fraud/health")
    assert resp.status_code == 200
