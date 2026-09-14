from __future__ import annotations

from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medireco.config import MEDICINE_DB_PATH
from medireco.predictor import model_info, predict_disease, predict_risk
from medireco.recommender import recommend_information

app = Flask(__name__)

REQUIRED = [
    "fever",
    "cough",
    "fatigue",
    "difficulty_breathing",
    "age",
    "gender",
    "blood_pressure",
    "cholesterol_level",
]
BINARY_FIELDS = ["fever", "cough", "fatigue", "difficulty_breathing"]


def normalize(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    out = dict(payload)

    for field in BINARY_FIELDS:
        value = str(out.get(field, "")).strip().lower()
        if value not in {"yes", "no"}:
            raise ValueError(f"{field} must be yes or no.")
        out[field] = value

    gender = str(out.get("gender", "")).strip().lower()
    if gender not in {"male", "female", "other"}:
        raise ValueError("gender must be male, female, or other.")
    out["gender"] = gender

    try:
        out["age"] = float(out["age"])
        out["blood_pressure"] = float(out["blood_pressure"])
        out["cholesterol_level"] = float(out["cholesterol_level"])
    except (TypeError, ValueError):
        raise ValueError("Age, blood pressure code, and cholesterol code must be numeric.")

    if not 0 <= out["age"] <= 120:
        raise ValueError("Age must be between 0 and 120.")
    if not 0 <= out["blood_pressure"] <= 10:
        raise ValueError("Blood pressure code must be between 0 and 10.")
    if not 0 <= out["cholesterol_level"] <= 10:
        raise ValueError("Cholesterol code must be between 0 and 10.")

    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "project": "Personalized Healthcare & Medicine Recommendation System",
            "model": "Random Forest Classifier",
        }
    )


@app.get("/api/model-info")
def api_model_info():
    return jsonify(model_info())


@app.get("/api/demo")
def api_demo():
    return jsonify(
        {
            "profiles": {
                "respiratory": {
                    "label": "Respiratory example",
                    "data": {
                        "age": 32,
                        "gender": "male",
                        "blood_pressure": 1,
                        "cholesterol_level": 1,
                        "fever": "yes",
                        "cough": "yes",
                        "fatigue": "yes",
                        "difficulty_breathing": "no",
                        "pregnancy": False,
                        "allergy": "",
                    },
                },
                "general": {
                    "label": "General example",
                    "data": {
                        "age": 25,
                        "gender": "female",
                        "blood_pressure": 0,
                        "cholesterol_level": 1,
                        "fever": "no",
                        "cough": "no",
                        "fatigue": "yes",
                        "difficulty_breathing": "no",
                        "pregnancy": False,
                        "allergy": "",
                    },
                },
            }
        }
    )


@app.post("/api/analyze")
def analyze():
    try:
        payload = request.get_json(force=True)
        normalized = normalize(payload)

        disease_predictions = predict_disease(normalized, top_k=3)
        risk = predict_risk(normalized)
        top_disease = disease_predictions[0]["disease"]

        candidates = recommend_information(
            MEDICINE_DB_PATH,
            top_disease,
            pregnancy=bool(payload.get("pregnancy", False)),
            allergy=str(payload.get("allergy", "")),
        )

        # User-entered context is kept separate from the model features.
        context = {
            "pregnancy": bool(payload.get("pregnancy", False)),
            "allergy": str(payload.get("allergy", "")).strip(),
        }

        return jsonify(
            {
                "disclaimer": (
                    "Educational decision support only. Not a diagnosis, prescription, "
                    "or substitute for professional medical care."
                ),
                "predictions": disease_predictions,
                "risk": risk,
                "medicine_information": candidates,
                "model_info": model_info(),
                "context": context,
            }
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
