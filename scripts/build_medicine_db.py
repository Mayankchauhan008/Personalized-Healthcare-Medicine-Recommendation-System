from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd


# Project root:
# Personalized_Healthcare/
ROOT = Path(__file__).resolve().parents[1]

# The original medicine database is optional.
# If it exists inside the project, it will be used.
POSSIBLE_SOURCES = [
    ROOT / "data" / "raw" / "medicine_database.pkl",
    ROOT / "data" / "medicine_database.pkl",
    ROOT / "medicine_database.pkl",
]

OUT = ROOT / "data" / "processed" / "medicine_knowledge.csv"


def strip_dose(text: str) -> str:
    """Remove dosage/frequency details from a medicine entry."""

    if " - " in text:
        text = text.split(" - ", 1)[0]

    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(mg|mcg|ml|%|puffs?|hours?|days?)\b",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(r"\s+", " ", text).strip(" -")

    return text


def find_database() -> Path | None:
    """Find medicine_database.pkl inside the project."""

    for path in POSSIBLE_SOURCES:
        if path.exists():
            return path

    return None


def build_fallback_database() -> pd.DataFrame:
    """
    Create a small safe reference database when the original pickle
    is not present.

    This is reference information only, not a prescription engine.
    """

    rows = [
        {
            "disease": "Fungal infection",
            "medicine": "Antifungal treatment",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Allergy",
            "medicine": "Antihistamine treatment",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Diabetes",
            "medicine": "Diabetes management",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Hypertension",
            "medicine": "Blood-pressure management",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Common Cold",
            "medicine": "Symptomatic cold care",
            "category": "supportive care",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Migraine",
            "medicine": "Migraine management",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Asthma",
            "medicine": "Asthma management",
            "category": "supportive care",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Pneumonia",
            "medicine": "Pneumonia management",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
        {
            "disease": "Heart disease",
            "medicine": "Cardiac care",
            "category": "reference",
            "information": "Reference category only; treatment should be selected by a qualified healthcare professional.",
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    source = find_database()

    if source is None:
        print("Original medicine_database.pkl was not found.")
        print("Creating fallback medicine reference database...")
        df = build_fallback_database()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.drop_duplicates().to_csv(OUT, index=False)
        print(f"Wrote {OUT} with {len(df)} rows")
        return

    print(f"Using medicine database: {source}")

    with source.open("rb") as file:
        db = pickle.load(file)

    rows = []

    for disease, payload in db.items():

        if not isinstance(payload, dict):
            continue

        medicines = payload.get("medicines", [])

        if not isinstance(medicines, list):
            continue

        for raw in medicines:

            clean = strip_dose(str(raw))

            if not clean:
                continue

            # Categorize the original project entries.
            if clean.startswith("💊"):
                category = "medicine"

            elif clean.startswith("💨"):
                category = "respiratory/supportive"

            elif clean.startswith("💉"):
                category = "injection"

            elif clean.startswith(("🧠", "🧘")):
                category = "therapy/supportive"

            elif clean.startswith(
                (
                    "📊",
                    "🍬",
                    "🍯",
                    "💧",
                    "🏃",
                    "🧂",
                )
            ):
                category = "self-care/monitoring"

            elif clean.startswith("🚨"):
                category = "emergency"

            else:
                category = "supportive care"

            clean = re.sub(r"^[^A-Za-z]+", "", clean)

            rows.append(
                {
                    "disease": disease,
                    "medicine": clean,
                    "category": category,
                    "information": (
                        "Reference item from the uploaded project "
                        "knowledge base; not a prescription."
                    ),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).drop_duplicates()

    if df.empty:
        print("No medicine records were found.")
        print("Creating fallback database instead.")
        df = build_fallback_database()

    df.to_csv(OUT, index=False)

    print(f"Wrote {OUT} with {len(df)} rows")


if __name__ == "__main__":
    main()