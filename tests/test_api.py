import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


def sample_payload():
    return {
        "age": 25,
        "gender": "female",
        "blood_pressure": 1,
        "cholesterol_level": 1,
        "fever": "no",
        "cough": "yes",
        "fatigue": "yes",
        "difficulty_breathing": "no",
        "pregnancy": False,
        "allergy": "",
    }


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_model_info():
    client = app.test_client()
    response = client.get("/api/model-info")
    assert response.status_code == 200
    assert response.get_json()["algorithm"] == "Random Forest Classifier"


def test_analyze():
    client = app.test_client()
    response = client.post("/api/analyze", json=sample_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["predictions"]
    assert body["risk"]["risk_level"]
