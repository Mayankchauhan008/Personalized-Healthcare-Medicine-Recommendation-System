from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

DATASET_PATH = DATA_RAW / "Cleaned_Dataset.csv"
MODEL_PATH = MODELS / "disease_risk_pipeline.joblib"
METRICS_PATH = MODELS / "metrics.json"
CLASSES_PATH = MODELS / "classes.json"
MEDICINE_DB_PATH = DATA_PROCESSED / "medicine_knowledge.csv"
