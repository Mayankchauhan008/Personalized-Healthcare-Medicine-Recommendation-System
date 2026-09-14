# Model Card

## Intended use
Educational demonstration of a supervised ML pipeline for condition ranking and risk-class prediction.

## Not intended for
Diagnosis, prescribing, dosage selection, emergency triage, or direct clinical use.

## Data
The supplied `Cleaned_Dataset.csv` contains 349 rows before deduplication and 300 after the project's cleaning step. It is highly imbalanced across disease labels.

## Model
Random Forest classifier with median imputation for numerical features, most-frequent imputation + one-hot encoding for categorical features, 500 trees, maximum depth 10, minimum leaf size 2.

## Target design
Eight relatively represented conditions are retained as explicit classes; all other disease labels become `Other`.

## Evaluation
The model is evaluated with a stratified 25% hold-out set. See `models/metrics.json` for the generated numbers.

A balanced metric is reported because raw accuracy is strongly affected by the large `Other` class.

## Limitations
- Very small dataset.
- Class imbalance.
- Limited input features.
- No clinical validation.
- No calibration or uncertainty guarantee.
- No demographic fairness study.
- Medicine content is retrieved from the uploaded project knowledge base and is not a validated prescribing guideline.
