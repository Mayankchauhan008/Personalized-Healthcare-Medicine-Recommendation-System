# Architecture

```text
Browser
  │
  ▼
Flask UI (/)
  │
  ▼
POST /api/analyze
  │
  ├── Input validation + normalization
  │
  ├── Disease Random Forest
  │      └── top-k probabilities
  │
  ├── Risk Random Forest
  │      └── class probabilities
  │
  └── Medicine information retrieval
         └── condition → reference items

Artifacts:
  data/raw/Cleaned_Dataset.csv
  data/processed/medicine_knowledge.csv
  models/*.joblib
  models/metrics.json
```

The project deliberately separates predictive ML from medication information retrieval. The predictive layer does not generate dosage instructions.
