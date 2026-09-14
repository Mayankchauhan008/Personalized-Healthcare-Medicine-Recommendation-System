from __future__ import annotations
import json, sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medireco.config import DATASET_PATH, MODELS
from medireco.preprocess import FEATURES, DISEASE_GROUPS, load_dataset

CAT = ['fever','cough','fatigue','difficulty_breathing','gender']
NUM = ['age','blood_pressure','cholesterol_level']


def make_pipeline():
    pre = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ]), CAT),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), NUM),
    ])
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=2,
        class_weight=None, random_state=42, n_jobs=-1
    )
    return Pipeline([('preprocess', pre), ('model', clf)])


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATASET_PATH)
    X = df[FEATURES]
    y = df['disease_group']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    disease_model = make_pipeline().fit(Xtr, ytr)
    disease_pred = disease_model.predict(Xte)

    risk_model = make_pipeline().fit(Xtr, df.loc[Xtr.index, 'risk_level'])
    risk_pred = risk_model.predict(Xte)

    metrics = {
        'disease_accuracy': float(accuracy_score(yte, disease_pred)),
        'disease_balanced_accuracy': float(balanced_accuracy_score(yte, disease_pred)),
        'risk_accuracy': float(accuracy_score(df.loc[Xte.index, 'risk_level'], risk_pred)),
        'dataset_rows': int(len(df)),
        'dataset_deduplicated_rows': int(len(df)),
        'disease_classes': sorted(map(str, disease_model.classes_)),
        'disease_report': classification_report(yte, disease_pred, output_dict=True, zero_division=0),
        'risk_report': classification_report(df.loc[Xte.index, 'risk_level'], risk_pred, output_dict=True, zero_division=0),
    }
    joblib.dump(disease_model, MODELS / 'disease_pipeline.joblib')
    joblib.dump(risk_model, MODELS / 'risk_pipeline.joblib')
    (MODELS / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (MODELS / 'classes.json').write_text(json.dumps(sorted(map(str, disease_model.classes_)), indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if 'report' not in k}, indent=2))

if __name__ == '__main__':
    main()
