"""Training / evaluation / artifact-saving for the Heart Disease Risk Classifier.

Pipeline (honest evaluation):
  1. load + clean + impute + feature-engineering (data_prep.py, features.py)
  2. 5-fold CV baseline  -> Logistic Regression (accuracy + recall)
  3. 5-fold CV grid search-> Random Forest (n_estimators, max_depth, min_samples_leaf)
  4. soft-voting ensemble (LR + RF) compared vs RF on accuracy AND recall
  5. decision-threshold picked from OUT-OF-FOLD train probabilities so that
     recall on real disease cases >= 90% (never tuned on the held-out test set)
  6. Low/Medium/High risk cutoffs derived from the OOF score distribution
  7. final model re-fit on full train set; artifacts written to artifacts/

Headline numbers are 5-fold CV plus a final held-out test evaluation.  No single
lucky train/test split is ever used for the headline number.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_prep
import features

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RECALL_TARGET = 0.90  # we want to catch >= 90% of real disease cases
RANDOM_STATE = 42
CV_FOLDS = 5

# Continuous columns that get standard-scaled (incl. engineered continuous ones).
CONTINUOUS_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "hr_fraction", "oldpeak_per_100hr", "n_risk_factors",
]

# Categorical columns fed through one-hot encoding.
CATEGORICAL_COLUMNS = data_prep.CATEGORICAL_COLUMNS

# Order expected by the models (raw + engineered features).
FEATURE_ORDER = (
    [c for c in data_prep.UCI_COLUMNS if c != "target"]
    + features.ADDED_FEATURES
)


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load, clean, impute and engineer features. Returns (X, y, report_map)."""
    raw = data_prep.load_heart_data(data_dir=PROJECT_DIR / "data")
    cleaned = data_prep.clean(raw)

    print("\n[data_prep] Missing values before imputation:")
    print(data_prep.missing_summary(cleaned.df).to_string())

    counts = cleaned.target.value_counts()
    ratio = counts.get(1, 0) / max(len(cleaned.target), 1)
    print(
        f"\n[data_prep] Class balance -> no-disease: {counts.get(0,0)}  "
        f"disease: {counts.get(1,0)}  (disease share {ratio:.1%})"
    )
    print(
        "[data_prep] Resampling: NOT applied. We use 5-fold CV + a recall-targeted "
        "threshold instead of SMOTE, keeping the pipeline simple and honest. "
        "If recall is still poor, resampling can be revisited."
    )

    X_clean = data_prep.impute_median(cleaned.df)
    X = features.build_features(X_clean)[FEATURE_ORDER]
    y = cleaned.target
    return X, y, cleaned.report_map


def make_preprocessor(X_train: pd.DataFrame):
    """ColumnTransformer: scale continuous, one-hot encode categorical."""
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]
    num_cols = [c for c in CONTINUOUS_COLUMNS if c in X_train.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def cv_scores(estimator, X, y, scoring=("accuracy", "recall")):
    """Run Stratified 5-fold CV and return {metric: [fold scores]}."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for metric in scoring:
        results[metric] = cross_val_score(estimator, X, y, cv=cv, scoring=metric, n_jobs=-1)
    return results


def summarize_cv(name, results):
    mean = {k: round(float(np.mean(v)), 4) for k, v in results.items()}
    print(f"\n=== {name} (5-fold CV) ===")
    for k, v in results.items():
        print(f"  {k}: {np.mean(v):.4f} (+/- {np.std(v):.4f})  folds={np.round(v,3).tolist()}")
    return mean


def make_lr(preprocessor):
    return Pipeline(steps=[("pre", preprocessor),
                           ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])


def make_rf(preprocessor, **rf_kwargs):
    rf_kwargs = {k.split('__', 1)[1] if k.startswith('clf__') else k: v for k, v in rf_kwargs.items()}
    return Pipeline(steps=[("pre", preprocessor),
                           ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **rf_kwargs))])


def make_ensemble(preprocessor, rf_kwargs=None):
    """Soft-voting LR + RF ensemble."""
    rf_kwargs = {k.split('__', 1)[1] if k.startswith('clf__') else k: v for k, v in (rf_kwargs or {}).items()}
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, **rf_kwargs)
    vote = VotingClassifier(estimators=[("lr", lr), ("rf", rf)], voting="soft", n_jobs=-1)
    return Pipeline(steps=[("pre", preprocessor), ("clf", vote)])


def grid_search_rf(preprocessor, X, y):
    """GridSearchCV over RF hyperparameters (5-fold CV, refit on recall)."""
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 8],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    rf = make_rf(preprocessor)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(rf, param_grid, cv=cv,
                      scoring={"recall": "recall", "accuracy": "accuracy"},
                      refit="recall", n_jobs=-1, return_train_score=True)
    gs.fit(X, y)
    print("\n=== Random Forest grid search (5-fold CV, refit on recall) ===")
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV recall: {gs.best_score_:.4f}")
    print(f"Best CV accuracy: {gs.cv_results_['mean_test_accuracy'][gs.best_index_]:.4f}")
    return gs.best_params_, gs


def run_baseline_rf_ensemble(X, y, preprocessor, best_rf_params):
    """Evaluate baseline LR, tuned RF and ensemble; return a metrics memo."""
    memo = {"X_shape": list(X.shape), "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS, "best_rf_params": best_rf_params}
    lr = make_lr(preprocessor)
    memo["baseline_logistic_regression"] = summarize_cv("Baseline Logistic Regression",
                                                         cv_scores(lr, X, y))
    rf = make_rf(preprocessor, **best_rf_params)
    memo["tuned_random_forest"] = summarize_cv("Tuned Random Forest (best grid params)",
                                               cv_scores(rf, X, y))
    ens = make_ensemble(preprocessor, rf_kwargs=best_rf_params)
    memo["lr_rf_soft_voting"] = summarize_cv("LR + RF soft-voting ensemble", cv_scores(ens, X, y))
    return memo


def choose_final_model(memo):
    """Pick RF vs ensemble: prefer higher recall unless it hurts accuracy badly."""
    rf = memo["tuned_random_forest"]
    en = memo["lr_rf_soft_voting"]
    rf_acc, rf_rec = rf["accuracy"], rf["recall"]
    en_acc, en_rec = en["accuracy"], en["recall"]
    print(f"\nModel selection -> RF(acc={rf_acc:.3f}, rec={rf_rec:.3f}) vs "
          f"Ensemble(acc={en_acc:.3f}, rec={en_rec:.3f})")
    if en_rec >= rf_rec + 0.005 and en_acc >= rf_acc - 0.02:
        print("Chosen: soft-voting ensemble (higher recall at no significant accuracy cost).")
        return "ensemble"
    print("Chosen: Random Forest (best balance; ensemble does not clearly improve).")
    return "random_forest"


def pick_threshold(oof_prob_pos, y, target=RECALL_TARGET):
    """Pick the HIGHEST decision threshold whose OOF recall >= target."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_t, best_rec, ok = None, 0.0, False
    for t in thresholds:
        rec = recall_score(y, (oof_prob_pos >= t).astype(int))
        if rec >= target:
            best_t, best_rec, ok = t, rec, True
    if not ok:
        for t in thresholds:
            rec = recall_score(y, (oof_prob_pos >= t).astype(int))
            if rec > best_rec:
                best_t, best_rec = t, rec
        print(f"[threshold] WARNING: cannot reach recall {target:.0%}; best {best_rec:.1%} at t={best_t}.")
        return float(best_t), float(best_rec), False
    print(f"[threshold] Chosen threshold = {best_t:.2f} -> OOF recall {best_rec:.2%}")
    return float(best_t), float(best_rec), True


def pick_risk_bands(oof_prob_pos, y, thresh):
    """Derive Low/Medium/High cutoffs from the OOF score distribution."""
    lo_cands, hi_cands = [25, 35, 45, 50], [55, 65, 75, 85]
    best = None
    for lo_p in lo_cands:
        for hi_p in hi_cands:
            if lo_p >= hi_p:
                continue
            lo_c = float(np.percentile(oof_prob_pos, lo_p))
            hi_c = float(np.percentile(oof_prob_pos, hi_p))
            if lo_c >= hi_c:
                continue
            low_mask = oof_prob_pos < lo_c
            high_mask = oof_prob_pos >= hi_c
            if low_mask.sum() == 0 or high_mask.sum() == 0:
                continue
            low_share = float(y[low_mask].mean())
            high_share = float(y[high_mask].mean())
            score = (1.0 - low_share) + high_share
            cand = (score, lo_c, hi_c, lo_p, hi_p, low_share, high_share)
            if best is None or score > best[0]:
                best = cand
    if best is None:
        lo_c, hi_c = 0.30, 0.70
        lo_p, hi_p, low_share, high_share, score = 0, 0, float("nan"), float("nan"), 0.0
    else:
        score, lo_c, hi_c, lo_p, hi_p, low_share, high_share = best
    cutoffs = sorted([round(float(lo_c), 3), round(float(hi_c), 3)])
    print(f"[risk-bands] Low<{cutoffs[0]:.3f}  Medium<{cutoffs[1]:.3f}  High>={cutoffs[1]:.3f}")
    print(f"[risk-bands] OOF percentiles low={lo_p} high={hi_p}; "
          f"Low-band disease share={low_share:.2%}, High-band disease share={high_share:.2%}")
    rationale = {
        "method": "percentile sweep on out-of-fold predicted probabilities",
        "low_percentile": int(lo_p),
        "high_percentile": int(hi_p),
        "cutoffs": cutoffs,
        "low_band_disease_share": None if np.isnan(low_share) else round(low_share, 3),
        "high_band_disease_share": None if np.isnan(high_share) else round(high_share, 3),
        "separation_score": round(float(score), 3),
        "threshold_for_disease": float(thresh),
    }
    return cutoffs, rationale


def build_final_estimator(preprocessor, chosen, best_rf_params):
    if chosen == "ensemble":
        return make_ensemble(preprocessor, rf_kwargs=best_rf_params)
    return make_rf(preprocessor, **best_rf_params)


def evaluate_on_test(model, X_test, y_test, threshold):
    prob = model.predict_proba(X_test)[:, 1]
    preds = (prob >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"\n=== Final model on held-out test set (threshold={threshold:.2f}) ===")
    print(f"  accuracy = {acc:.4f}   recall = {rec:.4f}\n  confusion matrix:\n{cm}")
    return {"accuracy": round(float(acc), 4), "recall": round(float(rec), 4), "threshold": float(threshold)}


def save_artifacts(model, preprocessor, X_train, X_test, y_train, y_test,
                   threshold, cutoffs, rationale, test_metrics, memo):
    try:
        encoded_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        encoded_names = None
    try:
        _bg = preprocessor.transform(X_train)
        shap_background = np.asarray(_bg)[:40]
    except Exception:
        shap_background = None
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "threshold": float(threshold),
        "risk_bands": {"low_cut": cutoffs[0], "high_cut": cutoffs[1]},
        "band_rationale": rationale,
        "input_columns": list(X_train.columns),
        "encoded_feature_names": encoded_names,
        "class_names": ["no_disease", "disease"],
        "class_balance_train": y_train.value_counts().to_dict(),
        "class_balance_test": y_test.value_counts().to_dict(),
        "report_groups": data_prep.REPORT_GROUPS,
        "shap_background": shap_background,
        "test_metrics": test_metrics,
    }
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print(f"[artifact] saved model.pkl ({os.path.getsize(ARTIFACT_DIR/'model.pkl')/1e3:.0f} KB)")

    metrics = {
        "test": test_metrics,
        "threshold": float(threshold),
        "risk_bands": {"low_cut": cutoffs[0], "high_cut": cutoffs[1]},
        "band_rationale": rationale,
        "cv_summary": {k: v for k, v in memo.items()
                       if k.startswith(("baseline", "tuned", "lr_rf"))},
        "best_rf_params": memo["best_rf_params"],
        "selected_model": memo["selected_model"],
        "note": ("5-fold CV and held-out test numbers reported as-is. "
                 "If accuracy < 90%, this is the honest figure (see README)."),
    }
    with open(ARTIFACT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[artifact] saved model_metrics.json")
    with open(ARTIFACT_DIR / "risk_cutoffs.json", "w") as f:
        json.dump({"risk_bands": {"low_cut": cutoffs[0], "high_cut": cutoffs[1]},
                   "rationale": rationale}, f, indent=2)
    print("[artifact] saved risk_cutoffs.json")


def main():
    X, y, report_map = prepare_data()
    print("\n[report-map] column -> report group mapping:")
    print(report_map.to_string(index=False))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    preprocessor = make_preprocessor(X_train)
    preprocessor.fit(X_train)
    print(f"\n[data] train rows: {len(X_train)}  test rows: {len(X_test)}")

    gs_best, gs = grid_search_rf(preprocessor, X_train, y_train)
    memo = run_baseline_rf_ensemble(X_train, y_train, preprocessor, gs_best)
    memo["selected_model"] = choose_final_model(memo)

    final = build_final_estimator(preprocessor, memo["selected_model"], gs_best)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, oof_recall, ok = pick_threshold(oof, y_train.to_numpy())
    cutoffs, rationale = pick_risk_bands(oof, y_train.to_numpy(), threshold)

    final.fit(X_train, y_train)
    test_metrics = evaluate_on_test(final, X_test, y_test, threshold)
    save_artifacts(final, preprocessor, X_train, X_test, y_train, y_test,
                   threshold, cutoffs, rationale, test_metrics, memo)

    print("\n=== DONE. Headline numbers ===")
    print(f"Accuracy (held-out test): {test_metrics['accuracy']:.2%}")
    print(f"Recall on disease cases  : {test_metrics['recall']:.2%}")
    print(f"Risk bands               : Low <{cutoffs[0]:.3f} | Medium <{cutoffs[1]:.3f} | High >={cutoffs[1]:.3f}")


if __name__ == "__main__":
    main()
