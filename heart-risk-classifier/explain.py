"""SHAP explainability for the Heart Disease Risk Classifier.

Loads the saved artifact (artifacts/model.pkl) and, for ANY single patient row,
returns:
  * the model's disease probability,
  * SHAP values aligned to human-readable feature labels,
  * the top contributing features with the direction each pushed the risk.

Works with the final model whether it is a Random Forest (TreeExplainer) or the
LR+RF soft-voting ensemble (KernelExplainer over a small background sample that
was stored in the artifact at training time).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from train import ARTIFACT_DIR  # reuse the artifact directory path

# Human-readable labels for the raw input columns.
RAW_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (0=female, 1=male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar >120 mg/dL",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved (bpm)",
    "exang": "Exercise-induced angina (1=yes)",
    "oldpeak": "ST depression (oldpeak, mm)",
    "slope": "ST slope",
    "ca": "Major vessels colored (0-3)",
    "thal": "Thalassemia result",
    "hr_fraction": "Achieved / predicted max HR fraction",
    "oldpeak_per_100hr": "ST depression per 100 bpm",
    "n_risk_factors": "Count of elevated risk factors",
}


def load_artifact(path: str | Path = None) -> dict:
    path = Path(path) if path else ARTIFACT_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found at {path}. Run `python train.py` first to train and save it."
        )
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def encoded_label(name: str) -> str:
    """Turn an encoded feature name (e.g. 'cat__cp_2.0') into a readable label."""
    if "__" not in name:
        return name
    kind, feat = name.split("__", 1)
    if kind == "num":
        return RAW_LABELS.get(feat, feat)
    # categorical one-hot: e.g. 'cp_2.0'
    if "_" in feat:
        base, val = feat.rsplit("_", 1)
        base_label = RAW_LABELS.get(base, base)
        return f"{base_label} = {val}"
    return RAW_LABELS.get(feat, feat)


def raw_value_for(row: pd.DataFrame, encoded: str):
    """Return the patient's raw value for an encoded feature."""
    if "__" not in encoded:
        return None
    kind, feat = encoded.split("__", 1)
    if kind == "num":
        return row[feat].iloc[0] if feat in row.columns else None
    base = feat.rsplit("_", 1)[0]
    return row[base].iloc[0] if base in row.columns else None


def make_explainer(artifact: dict):
    model = artifact["model"]
    clf = model.named_steps.get("clf")
    background = artifact.get("shap_background")
    if isinstance(clf, RandomForestClassifier):
        return shap.TreeExplainer(clf), "tree"
    # Ensemble -> KernelExplainer over stored background.
    if background is not None:
        predict_disease = lambda m: model.predict_proba(m)[:, 1]  # noqa: E731
        return shap.KernelExplainer(predict_disease, background), "kernel"
    # No background: fall back to explaining the Random Forest member.
    rf_member = clf.named_estimators_["rf"] if hasattr(clf, "named_estimators_") else clf.estimators_[0]
    return shap.TreeExplainer(rf_member), "tree-member"


def explain_patient(artifact: dict, row: dict, n_top: int = 6) -> dict:
    """Explain a single patient. `row` maps raw feature names -> values."""
    input_cols = artifact["input_columns"]
    df = pd.DataFrame([{c: row[c] for c in input_cols}], columns=input_cols)

    prob = artifact["model"].predict_proba(df)[:, 1][0]
    Xenc = artifact["preprocessor"].transform(df)

    explainer, kind = make_explainer(artifact)
    sv = explainer.shap_values(Xenc)
    expected = explainer.expected_value

    # Normalise to a 2D array (n_samples, n_features) for the disease class.
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    sv = sv.reshape(1, -1)

    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).ravel()[1] if len(expected) > 1 else np.asarray(expected).ravel()[0]

    encoded_names = artifact["encoded_feature_names"]
    features = []
    for i, name in enumerate(encoded_names):
        sh = float(sv[0, i])
        value = raw_value_for(df, name)
        direction = "increases risk" if sh > 0 else "decreases risk"
        features.append({
            "label": encoded_label(name),
            "shap": sh,
            "abs_shap": abs(sh),
            "value": value,
            "direction": direction,
        })

    features.sort(key=lambda f: f["abs_shap"], reverse=True)
    return {
        "prob_disease": float(prob),
        "base_value": float(expected),
        "explainer_kind": kind,
        "top_features": features[:n_top],
        "all_features": features,
    }


def summarize_in_words(res: dict) -> str:
    """Short human sentence built from the top SHAP drivers."""
    pushes_high = [f for f in res["top_features"] if f["shap"] > 0][:3]
    pushes_low = [f for f in res["top_features"] if f["shap"] < 0][:3]
    parts = []
    if pushes_high:
        parts.append("pushed toward Higher risk: " + ", ".join(f['label'] for f in pushes_high))
    if pushes_low:
        parts.append("pulled toward Lower risk: " + ", ".join(f['label'] for f in pushes_low))
    return "; ".join(parts) if parts else "no strongly contributing factors"
