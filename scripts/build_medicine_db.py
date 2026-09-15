from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POSSIBLE_SOURCES = [
    ROOT / "data" / "raw" / "medicine_database.pkl",
    ROOT / "data" / "medicine_database.pkl",
    ROOT / "medicine_database.pkl",
]
OUT = ROOT / "data" / "processed" / "medicine_knowledge.csv"


def clean_item(text: str) -> str:
    """Remove emoji and dose/frequency fragments from display names."""
    text = str(text).strip()
    if " - " in text:
        text = text.split(" - ", 1)[0]
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|%|puffs?|hours?|days?|times?)\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^[^A-Za-z]+", "", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def fallback() -> pd.DataFrame:
    rows = [
        ["Common Cold", "Symptomatic cold care", "Care guidance", "General supportive care reference."],
        ["Influenza", "Fluids and rest", "Care guidance", "General supportive care reference."],
        ["Asthma", "Asthma management", "Care guidance", "Use a clinician-developed asthma action plan."],
        ["Diabetes", "Blood glucose monitoring", "Care guidance", "Follow the monitoring plan provided by a healthcare professional."],
        ["Hypertension", "Blood-pressure monitoring", "Care guidance", "Monitor according to a clinician's plan."],
        ["Pneumonia", "Clinical assessment", "When to see doctor", "Breathing symptoms or worsening illness need professional review."],
        ["Heart disease", "Cardiac care", "Care guidance", "Treatment should be guided by a qualified clinician."],
        ["Migraine", "Migraine management", "Care guidance", "Keep a symptom/headache diary and seek appropriate medical advice."],
        ["Anxiety Disorders", "Mental-health support", "Care guidance", "Professional assessment and evidence-based support are recommended."],
        ["Depression", "Mental-health support", "Care guidance", "Professional assessment and evidence-based support are recommended."],
    ]
    return pd.DataFrame(rows, columns=["disease", "item", "category", "details"])


def main() -> None:
    source = next((p for p in POSSIBLE_SOURCES if p.exists()), None)

    if source is None:
        print("medicine_database.pkl not found; creating fallback reference database.")
        df = fallback()
    else:
        print(f"Using medicine database: {source}")
        with source.open("rb") as f:
            db = pickle.load(f)

        rows: list[list[str]] = []

        for disease, payload in db.items():
            if not isinstance(payload, dict):
                continue

            # Medicines: names only; do not export dose/frequency text.
            for raw in payload.get("medicines", []) or []:
                clean = clean_item(raw)
                if clean:
                    rows.append([
                        str(disease),
                        clean,
                        "Medicine reference",
                        "Reference item from the project knowledge base; suitability must be checked by a qualified professional.",
                    ])

            # General advice/support.
            for raw in payload.get("advice", []) or []:
                clean = clean_item(raw)
                if clean:
                    category = "Emergency" if any(w in clean.casefold() for w in ["emergency", "seek help", "urgent"]) else "Care guidance"
                    rows.append([
                        str(disease),
                        clean,
                        category,
                        "General care guidance from the project knowledge base; follow professional medical advice.",
                    ])

            # Foods are separated for a dedicated UI section.
            for raw in payload.get("foods_to_eat", []) or []:
                clean = clean_item(raw)
                if clean:
                    rows.append([str(disease), f"Consider: {clean}", "Food guidance", "Food guidance reference; individual needs vary."])

            for raw in payload.get("foods_to_avoid", []) or []:
                clean = clean_item(raw)
                if clean:
                    rows.append([str(disease), f"Limit/avoid: {clean}", "Food guidance", "Food guidance reference; individual needs vary."])

            doctor = str(payload.get("when_to_see_doctor", "")).strip()
            if doctor:
                rows.append([str(disease), doctor, "When to see doctor", "Seek professional medical review when this situation applies."])

        df = pd.DataFrame(rows, columns=["disease", "item", "category", "details"]).drop_duplicates()
        if df.empty:
            df = fallback()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
