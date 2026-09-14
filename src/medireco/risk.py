from __future__ import annotations
import joblib
from .config import ROOT

RISK_MODEL_PATH = ROOT / "models" / "risk_pipeline.joblib"

def predict_risk(payload: dict) -> dict:
    model = joblib.load(RISK_MODEL_PATH)
    row = [payload]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    return {
        "risk_level": str(pred),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}
    }
