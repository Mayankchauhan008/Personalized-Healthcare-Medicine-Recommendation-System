from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POSSIBLE_SOURCES = [ROOT / "data" / "raw" / "medicine_database.pkl", ROOT / "data" / "medicine_database.pkl", ROOT / "medicine_database.pkl"]
OUT = ROOT / "data" / "processed" / "medicine_knowledge.csv"


def strip_dose(text: str) -> str:
    if " - " in text:
        text = text.split(" - ", 1)[0]
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(mg|mcg|ml|%|puffs?|hours?|days?)\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" -")


def fallback() -> pd.DataFrame:
    return pd.DataFrame([
        {"disease":"Fungal infection","medicine":"Antifungal treatment","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Allergy","medicine":"Antihistamine treatment","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Diabetes","medicine":"Diabetes management","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Hypertension","medicine":"Blood-pressure management","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Common Cold","medicine":"Symptomatic cold care","category":"supportive care","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Migraine","medicine":"Migraine management","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Asthma","medicine":"Asthma management","category":"supportive care","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Pneumonia","medicine":"Pneumonia management","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
        {"disease":"Heart disease","medicine":"Cardiac care","category":"reference","information":"Reference category only; suitability should be reviewed by a qualified professional."},
    ])


def main() -> None:
    source = next((p for p in POSSIBLE_SOURCES if p.exists()), None)
    if source is None:
        print("medicine_database.pkl not found; creating fallback reference database.")
        df = fallback()
    else:
        print(f"Using medicine database: {source}")
        with source.open("rb") as f:
            db = pickle.load(f)
        rows = []
        for disease, payload in db.items():
            if not isinstance(payload, dict):
                continue
            for raw in payload.get("medicines", []):
                clean = re.sub(r"^[^A-Za-z]+", "", strip_dose(str(raw)))
                if not clean:
                    continue
                rows.append({"disease": disease, "medicine": clean, "category": "reference", "information": "Reference item from the uploaded project knowledge base; not a prescription."})
        df = pd.DataFrame(rows).drop_duplicates() if rows else fallback()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")


if __name__ == "__main__":
    main()
