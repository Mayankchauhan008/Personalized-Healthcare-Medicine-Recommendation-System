from __future__ import annotations
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import MODELS

@lru_cache(maxsize=1)
def load_disease_model():
    return joblib.load(MODELS / 'disease_pipeline.joblib')

@lru_cache(maxsize=1)
def load_risk_model():
    return joblib.load(MODELS / 'risk_pipeline.joblib')


def predict_disease(payload: dict, top_k: int = 3) -> list[dict]:
    model = load_disease_model()
    row = pd.DataFrame([payload])
    prob = model.predict_proba(row)[0]
    idx = np.argsort(prob)[::-1][:top_k]
    return [{'disease': str(model.classes_[i]), 'probability': round(float(prob[i]), 4)} for i in idx]


def predict_risk(payload: dict) -> dict:
    model = load_risk_model()
    row = pd.DataFrame([payload])
    prob = model.predict_proba(row)[0]
    idx = int(np.argmax(prob))
    return {
        'risk_level': str(model.classes_[idx]),
        'probabilities': {str(c): round(float(p),4) for c,p in zip(model.classes_, prob)},
    }
