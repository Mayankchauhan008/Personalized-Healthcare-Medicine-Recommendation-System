from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import CLASSES_PATH, METRICS_PATH, MODELS

FEATURE_LABELS = {
    "fever": "Fever",
    "cough": "Cough",
    "fatigue": "Fatigue",
    "difficulty_breathing": "Difficulty breathing",
    "age": "Age",
    "blood_pressure": "Blood pressure code",
    "cholesterol_level": "Cholesterol code",
    "gender": "Gender",
}


@lru_cache(maxsize=1)
def load_disease_model():
    return joblib.load(MODELS / "disease_pipeline.joblib")


@lru_cache(maxsize=1)
def load_risk_model():
    return joblib.load(MODELS / "risk_pipeline.joblib")


def _feature_importance(model) -> list[dict[str, Any]]:
    """Return aggregated Random Forest importance for original input features."""
    try:
        pre = model.named_steps["preprocess"]
        clf = model.named_steps["model"]
        raw_names = pre.get_feature_names_out()
        values = np.asarray(clf.feature_importances_, dtype=float)
        agg: dict[str, float] = {key: 0.0 for key in FEATURE_LABELS}

        for name, importance in zip(raw_names, values):
            clean = str(name).replace("cat__", "").replace("num__", "")
            matched = next((key for key in FEATURE_LABELS if clean == key or clean.startswith(key + "_")), None)
            if matched:
                agg[matched] += float(importance)

        total = sum(agg.values()) or 1.0
        rows = [
            {
                "feature": FEATURE_LABELS[key],
                "value": round((value / total) * 100, 1),
            }
            for key, value in agg.items()
        ]
        return sorted(rows, key=lambda x: x["value"], reverse=True)
    except Exception:
        return []


def predict_disease(payload: dict, top_k: int = 3) -> list[dict]:
    model = load_disease_model()
    row = pd.DataFrame([payload])
    prob = model.predict_proba(row)[0]
    idx = np.argsort(prob)[::-1][:top_k]
    return [
        {
            "rank": i + 1,
            "disease": str(model.classes_[j]),
            "probability": round(float(prob[j]), 4),
            "percentage": round(float(prob[j]) * 100, 1),
        }
        for i, j in enumerate(idx)
    ]


def predict_risk(payload: dict) -> dict:
    model = load_risk_model()
    row = pd.DataFrame([payload])
    prob = model.predict_proba(row)[0]
    idx = int(np.argmax(prob))
    return {
        "risk_level": str(model.classes_[idx]),
        "probabilities": {
            str(c): round(float(p), 4) for c, p in zip(model.classes_, prob)
        },
        "probability_percent": round(float(prob[idx]) * 100, 1),
    }


def model_info() -> dict:
    model = load_disease_model()
    try:
        metrics = __import__("json").loads(METRICS_PATH.read_text())
    except Exception:
        metrics = {}
    return {
        "algorithm": "Random Forest Classifier",
        "n_estimators": int(model.named_steps["model"].n_estimators),
        "classes": [str(x) for x in model.classes_],
        "feature_importance": _feature_importance(model),
        "metrics": {
            "disease_accuracy": metrics.get("disease_accuracy"),
            "disease_balanced_accuracy": metrics.get("disease_balanced_accuracy"),
            "risk_accuracy": metrics.get("risk_accuracy"),
            "dataset_rows": metrics.get("dataset_rows"),
        },
    }
