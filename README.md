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
