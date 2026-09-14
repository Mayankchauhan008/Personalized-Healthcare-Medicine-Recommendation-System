from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import app

def test_health_endpoint():
    c = app.test_client()
    r = c.get('/health')
    assert r.status_code == 200
    assert r.get_json()['status'] == 'ok'

def test_analyze_endpoint():
    c = app.test_client()
    payload = {
        'age': 25, 'gender': 'male', 'blood_pressure': 1, 'cholesterol_level': 1,
        'fever': 'yes', 'cough': 'yes', 'fatigue': 'yes', 'difficulty_breathing': 'no',
        'pregnancy': False, 'allergy': ''
    }
    r = c.post('/api/analyze', json=payload)
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert 'predictions' in body and len(body['predictions']) == 3
    assert 'risk' in body
