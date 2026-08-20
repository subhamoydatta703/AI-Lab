# Heart Disease Risk Classifier (prototype)

A teaching/demonstration prototype that predicts a patient's heart disease risk as
**Low / Medium / High** from values on their own medical reports, and explains *why*
the model reached that result using SHAP.

> **Not a diagnostic tool.** This gives no medical advice, does not read images/scans,
> and does not replace a doctor. It only outputs a risk level and an explanation.

## Stack

Python, pandas, NumPy, scikit-learn, SHAP, Streamlit, ucimlrepo. No other ML frameworks.

## Project layout

```
heart-risk-classifier/
├── data_prep.py      # load (ucimlrepo id=45, falls back to data/heart.csv), clean, report mapping, class balance
├── features.py       # small set of physiologically-justified engineered features
├── train.py          # 5-fold CV baseline, grid-search RF, ensemble compare, threshold + risk-band tuning, saves artifacts
├── explain.py        # SHAP explanation, reusable for any single patient
├── app.py            # Streamlit interface (form matches the three report groups)
├── requirements.txt
├── artifacts/        # model.pkl, model_metrics.json, risk_cutoffs.json (written by train.py)
└── data/             # optional local CSV fallback: heart.csv
```

## How to run

```bash
cd heart-risk-classifier
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1) Train + evaluate + save artifacts (needs internet once for the dataset, or place
#    a CSV at data/heart.csv with columns: age, sex, cp, trestbps, chol, fbs, restecg,
#    thalach, exang, oldpeak, slope, ca, thal, target)
python train.py

# 2) Launch the app (everything runs inside the Streamlit process; no backend)
streamlit run app.py
```

The dataset is the **UCI Heart Disease (Cleveland)** set, fetched at runtime via
`ucimlrepo` (id=45, 303 patients). If the network is unavailable the loader falls back
to a local CSV at `data/heart.csv`.

## Input form (matches the three report groups)

| Report group | Fields | Required? |
| --- | --- | --- |
| Lipid / Blood | cholesterol, fasting blood sugar | Required |
| ECG | resting ECG, max heart rate, ST depression (oldpeak), ST slope, exercise angina | Required |
| Echocardiography | major vessels colored (ca), thalassemia (thal) | Optional — improves confidence when provided |
| (Clinical) | age, sex, chest-pain type, resting blood pressure | Needed by the model |

The app notes that the echo report is optional; when omitted it uses typical values.

## Risk bands (derived from the score distribution, not arbitrary)

Cutoffs were chosen by a percentile sweep over **out-of-fold** predicted probabilities
(the model is never tuned on the held-out test set). Chosen values:

- **Low**: predicted disease probability `< 0.18`
- **Medium**: `0.18 – 0.86`
- **High**: `≥ 0.86`

These correspond to the 25th and 85th percentiles of the model's own output scores. In
the training distribution, the Low band contains only ~5% true-disease cases while the
High band contains ~97% — i.e. the bands separate the classes well. The separate
disease/no-disease decision threshold was tuned to **0.30** so that recall on real
disease cases is ≥ 90%.

## Results (actual, measured — 5-fold CV + held-out test)

| Model | Accuracy (5-fold CV) | Recall (5-fold CV) |
| --- | --- | --- |
| Logistic Regression (baseline) | 0.843 | 0.783 |
| Random Forest (grid-searched: 100 trees, no depth limit, leaf=1) | 0.831 | 0.792 |
| LR + RF soft-voting ensemble | 0.835 | 0.774 |

**Selected model: Random Forest** (best balance; the ensemble did not clearly improve it).

Threshold tuning picked **0.30** to reach ≥ 90% recall. On the **held-out test set**
(61 patients):

- Accuracy: **0.770**
- Recall: **1.000** (28/28 real disease cases caught; 14 healthy patients flagged as
  high — a conservative, catch-more trade-off).

### About the 90% accuracy target

**The 90% accuracy target was not reached.** The honest ceiling on this data is roughly
**83–84% accuracy** (RF / LR 5-fold CV) at the default threshold. Pushing recall to ≥ 90%
(lowering the threshold to 0.30) raised test recall to 100% but dropped test accuracy to
**77%** because of false positives — the classic accuracy/recall trade-off. We report
both numbers rather than quietly over-reporting accuracy.

**What limits it:** the dataset is small (303 patients) and the classes overlap in the
feature space (there is no clean decision boundary on these 13 measurements). CV fold
recall ranged roughly 0.68–0.87, showing high variance from the small sample. Resampling
(SMOTE) and feature selection were intentionally kept simple for this prototype; larger
or richer data would be the main lever to close the gap to 90%.

## Explanation (SHAP)

Every prediction comes with the top contributing factors and the direction each one
pushed the result, e.g. *"high cholesterol and ST depression pushed toward High risk."*
A bar chart of the top contributors is shown in the app.

## Notes / caveats

- This is a prototype: correctness and clarity were prioritised over engineering polish.
- Category values use the original UCI coding (chest pain 1–4, ST slope 1–3,
  thalassemia 3/6/7) — the form already matches this.
- Median imputation was used for the handful of missing `ca`/`thal` values.
