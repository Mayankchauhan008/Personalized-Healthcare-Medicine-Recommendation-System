from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medireco.config import DATASET_PATH, MEDICINE_DB_PATH


def test_dataset_exists():
    assert DATASET_PATH.exists()


def test_medicine_db_exists():
    assert MEDICINE_DB_PATH.exists()


def test_model_exists():
    assert (ROOT / "models" / "disease_pipeline.joblib").exists()
    assert (ROOT / "models" / "risk_pipeline.joblib").exists()
