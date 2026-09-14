from __future__ import annotations
from flask import Flask, jsonify, render_template, request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from medireco.config import MEDICINE_DB_PATH
from medireco.predictor import predict_disease, predict_risk
from medireco.recommender import recommend_information

app = Flask(__name__)

REQUIRED = ['fever','cough','fatigue','difficulty_breathing','age','gender','blood_pressure','cholesterol_level']

def normalize(payload: dict) -> dict:
    out = dict(payload)
    for c in ['fever','cough','fatigue','difficulty_breathing','gender']:
        out[c] = str(out.get(c, '')).strip().lower()
    for c in ['age','blood_pressure','cholesterol_level']:
        out[c] = float(out[c])
    return out

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/health')
def health():
    return jsonify({'status':'ok','project':'Personalized Healthcare & Medicine Recommendation System'})

@app.post('/api/analyze')
def analyze():
    try:
        payload = request.get_json(force=True)
        missing = [k for k in REQUIRED if k not in payload]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        normalized = normalize(payload)
        disease = predict_disease(normalized, top_k=3)
        risk = predict_risk(normalized)
        top_disease = disease[0]['disease']
        candidates = recommend_information(
            MEDICINE_DB_PATH, top_disease,
            pregnancy=bool(payload.get('pregnancy', False)),
            allergy=str(payload.get('allergy', '')),
        )
        return jsonify({
            'disclaimer': 'Educational decision support only. Not a diagnosis, prescription, or substitute for professional medical care.',
            'predictions': disease,
            'risk': risk,
            'medicine_information': candidates,
        })
    except (ValueError, TypeError) as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Unexpected server error: {exc}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
