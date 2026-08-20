"""Streamlit interface for the Heart Disease Risk Classifier.

Run:  streamlit run app.py   (from inside this folder, with the venv active)

Everything runs inside the Streamlit process - no separate backend.  The form
maps exactly to the three report groups the patient would have in hand, plus a
small demographics/clinical section that the UCI features need.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import data_prep  # noqa: F401  (kept for the report-group mapping reference)
import explain
import features

st.set_page_config(page_title="Heart Disease Risk", layout="centered")

st.title("Heart Disease Risk Classifier")
st.caption(
    "Prototype for teaching / demonstration only. Not a diagnostic tool - it does "
    "not replace a doctor, does not read scans, and gives no medical advice. "
    "It outputs a risk level and an explanation for that level."
)

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.pkl"


@st.cache_resource(show_spinner=False)
def get_artifact():
    if not ARTIFACT_PATH.exists():
        st.error(f"Trained model not found at {ARTIFACT_PATH}. Run `python train.py` first.")
        st.stop()
    return explain.load_artifact(ARTIFACT_PATH)


def cp_label(v):
    return {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}[v]


def slope_label(v):
    return {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[v]


def thal_label(v):
    return {3: "Normal", 6: "Fixed defect", 7: "Reversable defect"}[v]


def build_row(raw: dict) -> pd.DataFrame:
    """Add engineered features and return the exact column order the model expects."""
    art = get_artifact()
    df = features.build_features(pd.DataFrame([raw]))
    return df[[c for c in art["input_columns"]]]


def risk_level(prob, artifact) -> str:
    low = artifact["risk_bands"]["low_cut"]
    high = artifact["risk_bands"]["high_cut"]
    if prob < low:
        return "Low"
    if prob < high:
        return "Medium"
    return "High"


def main():
    art = get_artifact()
    risk = art["risk_bands"]
    st.info(
        "Enter values from your reports below. **Lipid and ECG reports are required**; "
        "the **Echocardiography report is optional** and improves confidence when provided."
    )

    with st.form("patient_form"):
        st.subheader("Clinical details")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.slider("Age (years)", 20, 100, 55)
        sex = c2.selectbox("Sex", ["Female", "Male"], index=1)
        cp = c3.selectbox("Chest pain type", options=[1, 2, 3, 4],
                          format_func=cp_label, index=3)
        trestbps = c4.number_input("Resting blood pressure (mmHg)", 80, 220, 130)

        st.subheader("Lipid / Blood Report (required)")
        l1, l2 = st.columns(2)
        chol = l1.number_input("Serum cholesterol (mg/dL)", 120, 600, 240)
        fbs = l2.selectbox("Fasting blood sugar >120 mg/dL", [0, 1],
                           format_func=lambda v: "No" if v == 0 else "Yes")

        st.subheader("ECG Report (required)")
        e1, e2, e3 = st.columns(3)
        restecg = e1.selectbox(
            "Resting ECG result", [0, 1, 2],
            format_func=lambda v: {0: "Normal", 1: "ST-T abnormal", 2: "LV hypertrophy"}[v],
        )
        thalach = e2.number_input("Max heart rate achieved (bpm)", 60, 230, 150)
        exang = e3.selectbox("Exercise-induced angina", [0, 1],
                             format_func=lambda v: "No" if v == 0 else "Yes")
        e4, e5 = st.columns(2)
        oldpeak = e4.number_input("ST depression (oldpeak, mm)", 0.0, 6.2, 1.0, 0.1)
        slope = e5.selectbox("ST slope", [1, 2, 3], format_func=slope_label)

        st.subheader("Echocardiography Report (optional)")
        echo_have = st.checkbox("I have my echocardiography values", value=False)
        x1, x2 = st.columns(2)
        ca_default, thal_default = 0, 3
        if echo_have:
            ca = x1.selectbox("Major vessels colored by fluoroscopy (0-3)", [0, 1, 2, 3], index=0)
            thal = x2.selectbox("Thalassemia result", [3, 6, 7],
                                format_func=thal_label, index=0)
        else:
            ca, thal = ca_default, thal_default
            st.caption(
                "Not provided -> using typical values (0 major vessels, normal thalassemia). "
                "Provided echo values improve confidence."
            )

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        raw = {
            "age": age, "sex": 1 if sex == "Male" else 0, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }
        X = build_row(raw)
        prob = float(art["model"].predict_proba(X)[:, 1][0])
        level = risk_level(prob, art)

        st.markdown("---")
        if level == "High":
            st.error(f"### Risk level: **{level}**")
        elif level == "Medium":
            st.warning(f"### Risk level: **{level}**")
        else:
            st.success(f"### Risk level: **{level}**")

        st.metric("Model disease probability", f"{prob:.1%}")
        st.caption(
            f"Risk bands from score distribution: Low < {risk['low_cut']:.2f} | "
            f"Medium < {risk['high_cut']:.2f} | High ≥ {risk['high_cut']:.2f}"
        )

        st.subheader("Why this result? (SHAP)")
        try:
            res = explain.explain_patient(art, dict(X.iloc[0]), n_top=8)
            st.write(explain.summarize_in_words(res))

            top = res["top_features"]
            labels = [f["label"] for f in top][::-1]
            shaps = [f["shap"] for f in top][::-1]
            colors = ["#c0392b" if s > 0 else "#2980b9" for s in shaps]

            fig, ax = plt.subplots(figsize=(7, max(3.0, 0.5 * len(top))))
            ax.barh(labels, shaps, color=colors, alpha=0.85)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_xlabel("SHAP value (toward disease risk)")
            ax.set_title(f"Top contributors - base rate {res['base_value']:.3f}")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not compute SHAP explanation: {exc}")

        st.caption("This is a demonstration prototype. See a doctor for any real concern.")


if __name__ == "__main__":
    main()
