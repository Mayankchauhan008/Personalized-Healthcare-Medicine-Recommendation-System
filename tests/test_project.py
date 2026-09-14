import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_exists():
    p = ROOT / 'data' / 'raw' / 'Cleaned_Dataset.csv'
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) > 100
    assert {'disease','fever','age','gender'}.issubset(df.columns)

def test_metrics_exist():
    metrics = json.loads((ROOT / 'models' / 'metrics.json').read_text())
    assert 0 <= metrics['disease_accuracy'] <= 1
    assert 0 <= metrics['risk_accuracy'] <= 1
