from __future__ import annotations
import pandas as pd

FEATURES = [
    'fever','cough','fatigue','difficulty_breathing',
    'age','gender','blood_pressure','cholesterol_level'
]
TARGET = 'disease_group'
DISEASE_GROUPS = ['Asthma','Bronchitis','Diabetes','Hypertension','Influenza','Migraine','Osteoporosis','Stroke']


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    required = ['disease','risk_level'] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    for col in ['fever','cough','fatigue','difficulty_breathing','gender']:
        df[col] = df[col].astype(str).str.strip().str.lower()
    for col in ['age','blood_pressure','cholesterol_level']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['age','blood_pressure','cholesterol_level','disease','risk_level']).copy()
    # Keep the eight well-represented conditions from the original project's model design;
    # collapse the long tail into an explicit Other class.
    df['disease_group'] = df['disease'].where(df['disease'].isin(DISEASE_GROUPS), 'Other')
    return df
