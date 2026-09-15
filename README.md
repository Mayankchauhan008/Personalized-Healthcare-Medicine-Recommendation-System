# Personalized Healthcare & Medicine Recommendation System

An interactive educational ML project built with Python, Flask, pandas and scikit-learn.

## Features

- Patient profile and symptom form with validation.
- Demo profiles for quick testing.
- Random Forest disease-class ranking.
- Dataset-derived risk-class prediction.
- Confidence/probability bars for top predictions.
- Aggregated Random Forest feature-importance dashboard.
- Reference medicine-information retrieval from the project knowledge base.
- Pregnancy/allergy context flags used only for information filtering.
- Downloadable JSON analysis report.
- Copy-result button.
- Dark-mode toggle.
- Responsive mobile/desktop layout.
- `/health`, `/api/model-info`, `/api/demo`, and `/api/analyze` endpoints.
- Training/evaluation artifacts in `models/`.

## Run on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\build_medicine_db.py
python scripts\train_all.py
python run.py
```

Open `http://127.0.0.1:5000`.

### Important
The original medicine pickle is optional. `scripts/build_medicine_db.py` no longer contains a hard-coded `/mnt/data/...` path. If the pickle is absent, a small reference-only fallback knowledge base is created so the application can still run.

## Project structure

```text
personalized_healthcare_ml/
├── app.py
├── run.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/medireco/
├── static/
├── templates/
└── tests/
```

## Safety / academic use

This project is an ML demonstration, not a clinical decision system. Predictions are based on the supplied training dataset and should not be interpreted as medical diagnosis or prescription.

## Evidence-linked medicine data
The medicine knowledge base was upgraded with source-linked records from official MedlinePlus/NIH pages discovered through web research. Each row includes source URL, evidence level, clinician-review requirement, pregnancy note and allergy note. The application retrieves these references for the top predicted classes and never invents a medicine when the knowledge base has no supported entry.

## PDF Check-up Report

The dashboard includes a `Download PDF report` button after a successful analysis.

The PDF contains:
- patient/check-up inputs
- selected symptoms and safety context
- top 3 model predictions and probabilities
- risk class and risk probabilities
- medicine/treatment references returned by the evidence-linked knowledge base
- care, lifestyle, urgent-care, and safety guidance when available
- source names and URLs
- Random Forest feature importance
- model metrics and dataset limitations
- educational-use disclaimer

The PDF is generated on demand by `src/medireco/report.py` and served through `POST /api/report`.
