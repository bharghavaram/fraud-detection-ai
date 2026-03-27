import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/fraud_xgboost.pkl")
    FRAUD_THRESHOLD: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    HIGH_RISK_THRESHOLD: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.8"))
    SHAP_ENABLED: bool = os.getenv("SHAP_ENABLED", "true").lower() == "true"
    LLM_REASONING_ENABLED: bool = os.getenv("LLM_REASONING_ENABLED", "true").lower() == "true"
    MAX_AMOUNT_THRESHOLD: float = float(os.getenv("MAX_AMOUNT_THRESHOLD", "10000.0"))
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

settings = Settings()
