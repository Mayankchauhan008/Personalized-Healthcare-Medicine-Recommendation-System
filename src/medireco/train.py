from __future__ import annotations

import json
import warnings

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CLASSES_PATH, DATASET_PATH, METRICS_PATH, MODEL_PATH
from .preprocess import FEATURES, TARGET, group_rare_diseases, load_dataset

CAT = ["fever", "cough", "fatigue", "difficulty_breathing", "gender"]
NUM = ["age", "blood_pressure", "cholesterol_level"]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", clf)])


def train() -> dict:
    warnings.filterwarnings("ignore")
    df = group_rare_diseases(load_dataset(DATASET_PATH), min_count=4)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    CLASSES_PATH.write_text(json.dumps(sorted(map(str, pipe.classes_)), indent=2))
    return metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps({k: v for k, v in m.items() if k != "classification_report"}, indent=2))
