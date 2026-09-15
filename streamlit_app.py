from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medireco.auth import authenticate, create_user, init_db
from medireco.config import MEDICINE_DB_PATH
from medireco.predictor import model_info, predict_disease, predict_risk
from medireco.recommender import recommend_information
from medireco.report import generate_checkup_pdf

st.set_page_config(
    page_title="MediReco | Personalized Healthcare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ---------- Styling ----------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f7fbff 0%, #eef5fb 100%);
    }
    [data-testid="stSidebar"] {
        background: #0b1f33;
    }
    [data-testid="stSidebar"] * {
        color: #f7fbff !important;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0b3558;
        margin-bottom: 0.15rem;
    }
    .brand-subtitle {
        color: #59728c;
        margin-bottom: 1.4rem;
    }
    .hero {
        background: linear-gradient(135deg, #0b4f7a 0%, #087f8c 100%);
        color: white;
        padding: 1.35rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.1rem;
        box-shadow: 0 12px 30px rgba(8, 80, 120, .16);
    }
    .hero h2 { margin: 0; font-size: 1.7rem; }
    .hero p { margin: .45rem 0 0; opacity: .9; }
    .metric-card {
        background: white;
        border: 1px solid #dce7f2;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(20, 55, 90, .06);
    }
    .prediction {
        background: white;
        border: 1px solid #dde8f3;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: .75rem;
    }
    .prediction-title { font-weight: 750; color: #123c61; }
    .small-muted { color: #6e8193; font-size: .88rem; }
    .success-box {
        background: #e9f9ef;
        color: #08753a;
        border: 1px solid #a8dfba;
        padding: .8rem 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-weight: 650;
    }
    .warning-box {
        background: #fff7e7;
        border: 1px solid #efd18c;
        color: #7c5a08;
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    .danger-box {
        background: #fff0f1;
        border: 1px solid #efb0b7;
        color: #a51e2a;
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    .section-title {
        color: #153d62;
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: .8rem;
        margin-bottom: .7rem;
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #dbe8f3;
        border-radius: 18px;
        padding: 1rem;
    }
    .footer-note {
        margin-top: 2rem;
        padding: 1rem;
        background: #eef5fb;
        border-radius: 12px;
        color: #60798f;
        font-size: .88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def login_page() -> None:
    st.markdown('<div class="brand-title">🩺 MediReco</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Personalized Healthcare & Medicine Recommendation System</div>', unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.3, 1])
    with center:
        st.markdown("## Welcome back")
        st.caption("Sign in to continue to your healthcare decision-support dashboard.")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
        if submitted:
            user = authenticate(email, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = {"id": user["id"], "full_name": user["full_name"], "email": user["email"]}
                st.rerun()
            else:
                st.error("Invalid email or password.")
        if st.button("Create a new account", use_container_width=True):
            st.session_state.auth_view = "register"
            st.rerun()


def register_page() -> None:
    st.markdown('<div class="brand-title">🩺 MediReco</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Create your personalized healthcare account</div>', unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.3, 1])
    with center:
        st.markdown("## Register")
        with st.form("register_form"):
            full_name = st.text_input("Full name", value="Adam", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True, type="primary")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(full_name, email, password)
                if ok:
                    st.markdown(f'<div class="success-box">✓ {message}</div>', unsafe_allow_html=True)
                    st.session_state.auth_view = "login"
                else:
                    st.error(message)
        if st.button("Back to login", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()


def sidebar() -> None:
    user = st.session_state.user
    with st.sidebar:
        st.markdown("## 🩺 MediReco")
        st.caption("ML-powered healthcare decision support")
        st.divider()
        st.markdown(f"**{user['full_name']}**")
        st.caption(user["email"])
        if st.button("Logout", use_container_width=True):
            for key in ["authenticated", "user", "analysis", "payload"]:
                st.session_state.pop(key, None)
            st.session_state.auth_view = "login"
            st.rerun()
        st.divider()
        info = model_info()
        st.metric("Model", "Random Forest")
        st.metric("Trees", info.get("n_estimators", "-"))
        m = info.get("metrics", {})
        if m.get("disease_accuracy") is not None:
            st.metric("Test accuracy", f"{float(m['disease_accuracy']) * 100:.1f}%")


def analysis_form() -> None:
    st.markdown('<div class="hero"><h2>Personalized Health Check</h2><p>Enter the available patient information. The model returns ranked classes, a risk estimate, and evidence-linked reference information.</p></div>', unsafe_allow_html=True)

    with st.form("health_form"):
        st.markdown('<div class="section-title">Patient profile</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            patient_name = st.text_input("Patient name", placeholder="Optional")
        with c2:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        with c3:
            gender = st.selectbox("Gender", ["male", "female", "other"])
        with c4:
            pregnancy = st.selectbox("Pregnancy flag", ["No", "Yes"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bp = st.slider("Blood pressure code", 0, 10, 5, help="Dataset coding scale, not mmHg.")
        with c2:
            chol = st.slider("Cholesterol code", 0, 10, 5, help="Dataset coding scale, not a lab result.")
        with c3:
            allergy = st.text_input("Known allergy / medicine", placeholder="e.g. penicillin")
        with c4:
            st.write("")
            st.caption("Context flags are used for filtering reference information only.")

        st.markdown('<div class="section-title">Symptoms</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            fever = st.checkbox("🌡️ Fever")
        with s2:
            cough = st.checkbox("😷 Cough")
        with s3:
            fatigue = st.checkbox("🔋 Fatigue")
        with s4:
            breathing = st.checkbox("🫁 Difficulty breathing")

        st.markdown('<div class="section-title">Quick demo profiles</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            demo_respiratory = st.form_submit_button("Use respiratory demo")
        with d2:
            demo_general = st.form_submit_button("Use general demo")
        with d3:
            analyze = st.form_submit_button("Analyze profile", type="primary")

    if demo_respiratory:
        st.session_state.demo_payload = {
            "fever": "yes", "cough": "yes", "fatigue": "yes", "difficulty_breathing": "yes",
            "age": 34, "gender": "male", "blood_pressure": 5, "cholesterol_level": 4,
            "pregnancy": False, "allergy": "",
        }
        st.rerun()
    if demo_general:
        st.session_state.demo_payload = {
            "fever": "no", "cough": "no", "fatigue": "yes", "difficulty_breathing": "no",
            "age": 52, "gender": "female", "blood_pressure": 7, "cholesterol_level": 7,
            "pregnancy": False, "allergy": "penicillin",
        }
        st.rerun()
    if analyze:
        payload = {
            "fever": "yes" if fever else "no",
            "cough": "yes" if cough else "no",
            "fatigue": "yes" if fatigue else "no",
            "difficulty_breathing": "yes" if breathing else "no",
            "age": int(age),
            "gender": gender,
            "blood_pressure": int(bp),
            "cholesterol_level": int(chol),
        }
        risk = predict_risk(payload)
        preds = predict_disease(payload, top_k=3)
        recs = recommend_information(
            MEDICINE_DB_PATH,
            preds,
            pregnancy=(pregnancy == "Yes"),
            allergy=allergy,
            risk_level=risk["risk_level"],
        )
        info = model_info()
        analysis = {
            "predictions": preds,
            "risk": risk,
            "recommendations": recs,
            "model_info": info,
        }
        pdf_payload = {
            **payload,
            "patient_name": patient_name,
            "pregnancy": pregnancy == "Yes",
            "allergy": allergy,
        }
        st.session_state.analysis = analysis
        st.session_state.payload = pdf_payload
        st.success("Analysis completed successfully.")


def demo_banner() -> None:
    demo = st.session_state.pop("demo_payload", None)
    if not demo:
        return
    st.info("Demo profile loaded. Click **Analyze profile** to run the model with the suggested values.")


def show_analysis() -> None:
    analysis = st.session_state.get("analysis")
    payload = st.session_state.get("payload")
    if not analysis:
        return

    st.markdown("## Analysis dashboard")
    predictions = analysis["predictions"]
    risk = analysis["risk"]
    info = analysis["model_info"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Top predicted class", predictions[0]["disease"])
    m2.metric("Top class probability", f"{predictions[0]['percentage']:.1f}%")
    m3.metric("Risk class", risk["risk_level"], f"{risk['probability_percent']:.1f}% probability")

    left, right = st.columns([1.1, .9])
    with left:
        st.markdown("### Top predictions")
        for p in predictions:
            st.markdown(f"**{p['rank']}. {p['disease']}** — {p['percentage']:.1f}%")
            st.progress(min(max(p["percentage"] / 100, 0), 1))
    with right:
        st.markdown("### Risk distribution")
        risk_df = pd.DataFrame({"Risk class": list(risk["probabilities"].keys()), "Probability": [v * 100 for v in risk["probabilities"].values()]})
        st.bar_chart(risk_df.set_index("Risk class"), y="Probability")

    st.markdown("### Patient check-up summary")
    patient_df = pd.DataFrame([
        ["Patient name", payload.get("patient_name") or "Not provided"],
        ["Age", payload.get("age")],
        ["Gender", payload.get("gender")],
        ["Blood pressure code", payload.get("blood_pressure")],
        ["Cholesterol code", payload.get("cholesterol_level")],
        ["Fever", payload.get("fever")],
        ["Cough", payload.get("cough")],
        ["Fatigue", payload.get("fatigue")],
        ["Difficulty breathing", payload.get("difficulty_breathing")],
        ["Pregnancy", "Yes" if payload.get("pregnancy") else "No"],
        ["Known allergy / medicine", payload.get("allergy") or "None provided"],
    ], columns=["Field", "Value"])
    st.dataframe(patient_df, hide_index=True, use_container_width=True)

    st.markdown("### Treatment & care reference")
    recommendations = analysis["recommendations"]
    st.caption(recommendations.get("message", "Evidence-linked reference options."))
    groups = recommendations.get("groups", [])
    if not groups:
        st.warning("No evidence-linked reference information was found for the returned predictions.")
    for group in groups:
        with st.container(border=True):
            st.markdown(f"#### {group['disease']} — {group['probability']:.1f}%")
            tabs = st.tabs(["Medicine / treatment", "Guidance", "Safety & urgent care"])
            with tabs[0]:
                meds = group.get("medicines", [])
                if not meds:
                    st.info("No medicine-specific reference was returned for this class.")
                for med in meds:
                    st.markdown(f"**{med['item']}**")
                    st.write(med["details"])
                    st.caption(f"{med['source_name']} • {med['evidence_level']}")
                    if med.get("source_url"):
                        st.markdown(f"[Open source reference]({med['source_url']})")
                    st.divider()
            with tabs[1]:
                for section in ["support", "food"]:
                    for item in group.get(section, []):
                        st.markdown(f"**{item['item']}** — {item['details']}")
                if not group.get("support") and not group.get("food"):
                    st.info("No additional guidance was returned.")
            with tabs[2]:
                for section in ["when_to_seek_care", "emergency", "safety"]:
                    for item in group.get(section, []):
                        if section == "emergency":
                            st.error(f"**{item['item']}** — {item['details']}")
                        else:
                            st.warning(f"**{item['item']}** — {item['details']}")
                if not any(group.get(k) for k in ["when_to_seek_care", "emergency", "safety"]):
                    st.info("No additional urgent/safety item was returned.")

    if info.get("feature_importance"):
        st.markdown("### Model explainability")
        imp = pd.DataFrame(info["feature_importance"][:8])
        imp = imp.rename(columns={"feature": "Feature", "value": "Importance (%)"}).set_index("Feature")
        st.bar_chart(imp)

    st.markdown("### Download report")
    pdf_bytes = generate_checkup_pdf(payload, analysis)
    st.download_button(
        "📄 Download PDF check-up report",
        data=pdf_bytes,
        file_name="medireco-checkup-report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        '<div class="footer-note"><b>Use this responsibly.</b> Educational decision support only. '
        'Not a diagnosis, prescription, dosage instruction, or substitute for professional medical care.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    if not st.session_state.authenticated:
        if st.session_state.auth_view == "register":
            register_page()
        else:
            login_page()
        return

    sidebar()
    demo_banner()
    analysis_form()
    show_analysis()


if __name__ == "__main__":
    main()
