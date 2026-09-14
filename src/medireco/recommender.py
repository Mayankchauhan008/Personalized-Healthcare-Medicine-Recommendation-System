from __future__ import annotations
from pathlib import Path
import pandas as pd

MEDICINE_COLUMNS = ['disease','medicine','category','information']

def load_medicine_db(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[MEDICINE_COLUMNS].copy()

def recommend_information(path: str | Path, disease: str, pregnancy: bool = False, allergy: str = '') -> list[dict]:
    db = load_medicine_db(path)
    subset = db[db['disease'].str.casefold() == disease.casefold()].copy()
    if subset.empty:
        return []
    note = 'Reference information only. Verify suitability with a licensed clinician or pharmacist.'
    if pregnancy:
        note = 'Pregnancy selected: medication safety requires clinician/pharmacist review.'
    allergy = allergy.strip().casefold()
    out = []
    for _, row in subset.iterrows():
        item_note = note
        if allergy and allergy in str(row['medicine']).casefold():
            item_note = 'Possible allergy-name match: do not use without professional review.'
        out.append({
            'medicine': row['medicine'],
            'category': row['category'],
            'information': row['information'],
            'safety_note': item_note,
        })
    return out
