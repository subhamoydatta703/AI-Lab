"""Feature engineering for the Heart Disease Risk Classifier (prototype).

Adds a SMALL number of physiologically-sensible combined features on top of the
13 raw UCI columns.  We deliberately keep this tiny (nothing like hundreds of
auto-generated interactions) so every added column is defensible and explainable.

Raw numeric/continuous columns are left as-is; this module only ADDS columns.
The caller splits categorical vs continuous downstream.

Added features
--------------
* hr_fraction      = thalach / max(1, (220 - age))
    Fraction of the age-predicted maximum heart rate that the patient actually
    achieved during exercise.  A low value suggests chronotropic intolerance
    (the heart can't reach the expected rate under load) -> higher risk.
    This is the age x max-heart-rate interaction requested in the spec.

* oldpeak_per_100hr  = oldpeak * 100 / max(1, thalach)
    ST-segment depression (a sign of myocardial ischaemia during exercise)
    expressed per unit of achieved heart rate.  Depression observed at a lower
    exercise heart rate is more worrying, so this folds both columns together.

* n_risk_factors     = count of elevated classic risk factors among:
                       fbs>0, chol>240, trestbps>140, exang==1, oldpeak>0
    A simple, readable "burden of risk factors" tally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Names of the engineered (added) columns.
ADDED_FEATURES: list[str] = ["hr_fraction", "oldpeak_per_100hr", "n_risk_factors"]

# Without scaling, logistic regression needs similar scales; adding these
# continuous features keeps everything on comparable-ish ranges already, but
# the pipeline still scales continuous features downstream (see train.py).


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered columns appended.

    ``df`` must contain the raw numeric columns: age, thalach, oldpeak, chol,
    trestbps, fbs, exang.
    """
    out = df.copy()

    # 1) fraction of age-predicted max HR achieved
    max_hr_pred = np.maximum(220.0 - out["age"], 1.0)
    out["hr_fraction"] = out["thalach"] / max_hr_pred

    # 2) ST depression per unit of achieved heart rate (x100 for readability)
    out["oldpeak_per_100hr"] = out["oldpeak"] * 100.0 / np.maximum(out["thalach"], 1.0)

    # 3) burden of classic CVD risk factors
    out["n_risk_factors"] = (
        (out["fbs"] >= 1).astype(int)
        + (out["chol"] >= 240).astype(int)
        + (out["trestbps"] > 140).astype(int)
        + (out["exang"] == 1).astype(int)
        + (out["oldpeak"] > 0.0).astype(int)
    )

    return out