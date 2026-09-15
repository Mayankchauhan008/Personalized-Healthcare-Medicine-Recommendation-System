from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

COLUMNS = [
    "disease", "item", "category", "details", "source_name", "source_url",
    "evidence_level", "requires_clinician", "pregnancy_note", "allergy_note"
]

# Evidence-linked fallback references. These are ONLY used when the local
# knowledge base has no medicine/treatment item for the predicted class.
# They are reference options to discuss with a clinician/pharmacist, not
# automatic prescriptions and contain no dosage instructions.
FALLBACK_MEDICINES: dict[str, list[dict[str, str]]] = {
    "asthma": [
        {
            "item": "Albuterol (short-acting bronchodilator)",
            "details": "A quick-relief bronchodilator used for wheezing, shortness of breath, coughing and chest tightness in asthma. Use only according to an existing clinician treatment plan.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/druginfo/meds/a682145.html",
        },
        {
            "item": "Inhaled corticosteroid",
            "details": "A controller medicine class used to reduce airway inflammation and prevent asthma symptoms. The specific medicine and plan should be selected with a healthcare professional.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/patientinstructions/000005.htm",
        },
    ],
    "influenza": [
        {
            "item": "Oseltamivir (antiviral)",
            "details": "An antiviral used for some influenza infections, especially when treatment is started early. A clinician should determine whether antiviral treatment is appropriate.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/druginfo/meds/a699040.html",
        },
        {
            "item": "Symptom-relief medicines",
            "details": "Some people use medicines for fever, aches or other symptoms. Product choice depends on age, health conditions, allergies and other medicines; check with a pharmacist or clinician.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/article/000080.htm",
        },
    ],
    "chronic obstructive pulmonary disease (copd)": [
        {
            "item": "Short-acting bronchodilator",
            "details": "Quick-relief bronchodilator therapy may open the airways during symptoms or flare-ups. The specific inhaler should be selected with a clinician.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/patientinstructions/000026.htm",
        },
        {
            "item": "Long-acting inhaler therapy",
            "details": "COPD may require daily controller medicines such as long-acting bronchodilators and selected inhaled therapies, based on clinical assessment.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/patientinstructions/000025.htm",
        },
    ],
    "diabetes": [
        {
            "item": "Metformin",
            "details": "A commonly used medicine for type 2 diabetes. Suitability depends on the individual, including kidney and liver health and other factors.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/druginfo/meds/a696005.html",
        },
        {
            "item": "Other glucose-lowering medicines",
            "details": "Several medicine classes may be used alone or in combination depending on glucose control, kidney or heart health, side effects and other factors.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/patientinstructions/000989.htm",
        },
    ],
    "hypertension": [
        {
            "item": "ACE inhibitor",
            "details": "One medicine class used to lower blood pressure. Selection depends on other health conditions and clinician assessment.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/bloodpressuremedicines.html",
        },
        {
            "item": "ARB / calcium-channel blocker / diuretic",
            "details": "These are common antihypertensive medicine classes. The most appropriate option depends on the patient and should be selected by a clinician.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/bloodpressuremedicines.html",
        },
    ],
    "pneumonia": [
        {
            "item": "Antibiotic therapy when bacterial",
            "details": "Bacterial pneumonia may require antibiotics selected by a clinician. Antibiotics do not treat viral pneumonia.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/pneumonia.html",
        },
        {
            "item": "Supportive care",
            "details": "Hydration, rest and symptom management may be part of care; the treatment plan depends on the cause and severity.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/pneumonia.html",
        },
    ],
    "migraine": [
        {
            "item": "Migraine-specific acute treatment",
            "details": "Prescription medicines may be used to stop migraine attacks. Choice depends on symptoms, health conditions and other medicines.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/migraine.html",
        },
        {
            "item": "Pain-relief medicine",
            "details": "Some people use pain-relief medicines for migraine. A clinician or pharmacist can help determine an appropriate option and avoid overuse.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/migraine.html",
        },
    ],
    "hypothyroidism": [
        {
            "item": "Levothyroxine",
            "details": "Thyroid hormone replacement commonly used for hypothyroidism. The dose and monitoring plan require clinician supervision.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/druginfo/meds/a682461.html",
        },
    ],
    "gout": [
        {
            "item": "Anti-inflammatory treatment",
            "details": "NSAIDs, colchicine or corticosteroids may be used for gout flares depending on the patient and clinician assessment.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/article/000422.htm",
        },
        {
            "item": "Urate-lowering therapy",
            "details": "Long-term urate-lowering medicines may be used for some people with gout to reduce future attacks.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/article/000422.htm",
        },
    ],
    "rheumatoid arthritis": [
        {
            "item": "DMARD therapy",
            "details": "Disease-modifying antirheumatic drugs such as methotrexate are used in rheumatoid arthritis under specialist supervision and monitoring.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/article/000431.htm",
        },
    ],
    "urinary tract infection": [
        {
            "item": "Antibiotic therapy when bacterial UTI is confirmed",
            "details": "UTIs are commonly treated with antibiotics selected according to symptoms, testing, allergies, pregnancy status and local resistance patterns.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/ency/article/000521.htm",
        },
    ],
    "depression": [
        {
            "item": "Antidepressant medication",
            "details": "Antidepressants are one treatment option for depression and are selected based on the individual's symptoms, health history and other medicines.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/depression.html",
        },
        {
            "item": "Psychotherapy",
            "details": "Psychotherapy is another evidence-based treatment option and may be used alone or together with medication.",
            "source_name": "MedlinePlus",
            "source_url": "https://medlineplus.gov/depression.html",
        },
    ],
}


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().casefold(): c for c in df.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    return None


def load_knowledge_base(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {csv_path}")
    raw = pd.read_csv(csv_path).fillna("")
    disease_col = _first_existing(raw, ["disease", "condition", "diagnosis", "class"])
    item_col = _first_existing(raw, ["item", "medicine", "medication", "recommendation", "name", "title"])
    if not disease_col or not item_col:
        raise ValueError("Knowledge base requires disease + item/medicine columns.")

    result = pd.DataFrame()
    result["disease"] = raw[disease_col].astype(str).str.strip()
    result["item"] = raw[item_col].astype(str).str.strip()
    for target, aliases, default in [
        ("category", ["category", "type", "kind"], "Reference"),
        ("details", ["details", "information", "description", "note", "notes"], "Reference information."),
        ("source_name", ["source_name", "source", "publisher"], "Project knowledge base"),
        ("source_url", ["source_url", "url", "source_link"], ""),
        ("evidence_level", ["evidence_level", "evidence"], "Reference"),
        ("requires_clinician", ["requires_clinician", "clinical_review"], "Clinical review recommended"),
        ("pregnancy_note", ["pregnancy_note"], "Pregnancy/breastfeeding requires clinician review."),
        ("allergy_note", ["allergy_note"], "Check active ingredients and allergy history before use."),
    ]:
        col = _first_existing(raw, aliases)
        result[target] = raw[col].astype(str).str.strip() if col else default

    def norm(cat: str, item: str) -> str:
        c = cat.casefold().strip()
        i = item.casefold().strip()
        if c in {"medicine", "medication", "medicine reference", "drug", "prescription"}:
            return "Medicine reference"
        if any(x in c for x in {"medicine", "medication", "drug", "antiviral", "antibiotic", "bronchodilator", "controller"}):
            return "Medicine reference"
        if c in {"care guidance", "supportive care", "care", "self-care / support"}: return "Care guidance"
        if c in {"emergency", "urgent"}: return "Emergency"
        if c in {"safety"}: return "Safety"
        if "doctor" in c or "clinical" in c: return "When to see doctor"
        # Some older data stores the medicine type in the item only.
        if any(x in i for x in {"albuterol", "inhaled corticosteroid", "oseltamivir", "metformin", "levothyroxine", "antibiotic", "antiviral", "ace inhibitor", "arb", "calcium channel blocker", "diuretic", "dmard", "antidepressant", "anti-inflammatory"}):
            return "Medicine reference"
        return cat.strip() or "Reference"

    result["category"] = [norm(str(c), str(i)) for c, i in zip(result["category"], result["item"])]
    result = result[(result["disease"] != "") & (result["item"] != "")].drop_duplicates()
    return result[COLUMNS].reset_index(drop=True)


def _matches_prediction(db: pd.DataFrame, disease: str) -> pd.DataFrame:
    key = disease.casefold().strip()
    exact = db[db["disease"].str.casefold() == key]
    if not exact.empty:
        return exact
    return db[db["disease"].str.casefold().map(lambda x: key in x or x in key)]


def _fallback_for_disease(disease: str) -> list[dict[str, str]]:
    key = disease.casefold().strip()
    # Exact and normalized matches first.
    if key in FALLBACK_MEDICINES:
        return FALLBACK_MEDICINES[key]
    for name, entries in FALLBACK_MEDICINES.items():
        if key in name or name in key:
            return entries
    return []


def recommend_information(
    path: str | Path,
    predictions: list[dict[str, Any]],
    pregnancy: bool = False,
    allergy: str = "",
    risk_level: str = "",
) -> dict[str, Any]:
    db = load_knowledge_base(path)
    allergy_text = allergy.strip().casefold()
    high_risk = risk_level.casefold() in {"high", "critical"}
    groups = []

    for pred in predictions:
        disease = str(pred.get("disease", "")).strip()
        subset = _matches_prediction(db, disease)

        bucket = {"medicines": [], "support": [], "food": [], "when_to_seek_care": [], "emergency": [], "safety": []}
        for _, row in subset.iterrows():
            item, category, details = map(str, [row["item"], row["category"], row["details"]])
            low = f"{item} {details}".casefold()
            safety = str(row["requires_clinician"]) or "Clinical review recommended"
            if allergy_text and allergy_text in low:
                safety = "Possible allergy-name match. Do not use this item until a pharmacist/clinician reviews it."
            elif pregnancy and category == "Medicine reference":
                safety = str(row["pregnancy_note"])
            record = {
                "item": item,
                "category": category,
                "details": details,
                "safety_note": safety,
                "source_name": str(row["source_name"]),
                "source_url": str(row["source_url"]),
                "evidence_level": str(row["evidence_level"]),
            }
            if category == "Emergency": bucket["emergency"].append(record)
            elif category == "Medicine reference": bucket["medicines"].append(record)
            elif category == "When to see doctor": bucket["when_to_seek_care"].append(record)
            elif category == "Safety": bucket["safety"].append(record)
            elif "food" in category.casefold() or "diet" in category.casefold(): bucket["food"].append(record)
            else: bucket["support"].append(record)

        # Critical fix: if the database only has care/support items for a class,
        # add evidence-linked medicine reference options from the fallback map.
        if not bucket["medicines"]:
            for ref in _fallback_for_disease(disease):
                safety_note = "Reference option only — discuss with a qualified clinician/pharmacist; no dosage is provided."
                if pregnancy:
                    safety_note = "Pregnancy/breastfeeding can change treatment choices. Ask a clinician before using any medicine."
                if allergy_text and allergy_text in f"{ref['item']} {ref['details']}".casefold():
                    safety_note = "Possible allergy-name match. Do not use this item until a pharmacist/clinician reviews it."
                bucket["medicines"].append({
                    "item": ref["item"],
                    "category": "Medicine reference",
                    "details": ref["details"],
                    "safety_note": safety_note,
                    "source_name": ref["source_name"],
                    "source_url": ref["source_url"],
                    "evidence_level": "Official patient/medical reference",
                })

        if high_risk and not bucket["emergency"]:
            bucket["when_to_seek_care"].insert(0, {
                "item": "Prompt clinical review",
                "category": "When to see doctor",
                "details": "The model risk class is high; use the result to support prompt professional assessment rather than self-treatment.",
                "safety_note": "This project does not prescribe treatment.",
                "source_name": "MediReco safety policy",
                "source_url": "",
                "evidence_level": "Project safety rule",
            })

        # Keep groups that have either database support or a fallback medicine reference.
        if subset.empty and not bucket["medicines"]:
            continue

        groups.append({
            "disease": disease,
            "probability": pred.get("percentage", 0),
            "medicines": bucket["medicines"][:8],
            "support": bucket["support"][:10],
            "food": bucket["food"][:8],
            "when_to_seek_care": bucket["when_to_seek_care"][:5],
            "emergency": bucket["emergency"][:5],
            "safety": bucket["safety"][:5],
        })

    return {
        "matched": bool(groups),
        "message": (
            "Evidence-linked reference options for the top predicted classes. These are not prescriptions or dosing instructions."
            if groups else
            "No evidence-linked reference information was found for the returned predictions."
        ),
        "groups": groups[:3],
    }
