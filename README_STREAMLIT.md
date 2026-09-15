# MediReco - Streamlit Deployment

This is the Streamlit version of the Personalized Healthcare & Medicine Recommendation System.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/build_medicine_db.py
python scripts/train_all.py
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload this project.
2. Push the complete repository, including `data/`, `models/`, `src/`, `.streamlit/`, `streamlit_app.py`, and `requirements.txt`.
3. Go to Streamlit Community Cloud and choose **Create app**.
4. Select the repository and branch.
5. Set the entrypoint to `streamlit_app.py`.
6. Choose Python 3.12 (or use the same Python version you test locally).
7. Deploy.

### Important note about local SQLite authentication

The included register/login system uses a local SQLite database under `instance/`. This is suitable for a student/demo deployment, but Streamlit Community Cloud storage is not a durable production database. For a public production application, replace SQLite with a persistent service such as PostgreSQL/Supabase/Firebase, or use Streamlit's supported OAuth/OIDC authentication.

## Main features

- Register/login with hashed passwords
- Interactive patient profile form
- Demo patient profiles
- Random Forest disease prediction
- Risk probability visualization
- Evidence-linked treatment/care references
- Medicine source links
- Feature importance / explainability
- PDF check-up report download
- No visible API section
- No JSON download button

## Medical safety

This is an educational ML decision-support project. It is not a diagnostic or prescribing system. Medicine entries are reference options and do not include dosage instructions.
