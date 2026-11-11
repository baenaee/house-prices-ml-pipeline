"""B1: register the four baseline experiments E001-E004 on feature set "base".

    python experiments/models/run_baselines.py

All use keep_all rows and shuffled KFold(5, seed 42). They set the reference levels every
later experiment is compared with: no model (mean), an unregularised linear model on
numbers only, a regularised linear model on everything, and an untuned booster.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import run_and_register  # noqa: E402

FEATURE_SET = "base"
MEAN_RMSE_REF = {}  # filled while running, so later interpretations can quote the mean baseline


def _interp_mean(r):
    MEAN_RMSE_REF["mean"] = r.mean
    return (f"RMSE {r.mean:.4f}: this is roughly the standard deviation of log-price, the floor "
            "any model must beat. Anything near it has learned nothing.", "retain")


def _interp_ols(r):
    gain = 1 - r.mean / MEAN_RMSE_REF["mean"]
    return (f"RMSE {r.mean:.4f} (std {r.std:.4f}), {gain:.0%} below the mean baseline. Numeric columns "
            "alone carry most of the signal; unregularised OLS on correlated areas is the benchmark "
            "regularisation must beat. A large fold std would flag instability from collinearity.", "retain")


def _interp_ridge(r):
    return (f"RMSE {r.mean:.4f} (std {r.std:.4f}). First model using categorical variables via one-hot "
            "and shrinkage; compare with OLS-numeric to see what categories plus regularisation add. "
            "alpha=10 is an untuned guess; the grid in run_linear_family.py refines it.", "retain")


def _interp_xgb(r):
    return (f"RMSE {r.mean:.4f} (std {r.std:.4f}) with library defaults. Gives the untuned booster level; "
            "tuning (B3) and outlier policy (B2) are judged against it.", "retain")


EXPERIMENTS = [
    dict(
        exp_id="E001", title="Mean baseline",
        hypothesis="Predicting the training mean of log-price scores worst; every real model must beat it.",
        rationale="Provides the zero-skill reference so improvements can be expressed as relative gains.",
        implementation="models.build_mean_baseline: DummyRegressor(strategy='mean'); features: base.",
        builder="houseprices.models:build_mean_baseline", builder_kwargs={}, interpret=_interp_mean, tags=["baseline"],
    ),
    dict(
        exp_id="E002", title="OLS on numeric columns",
        hypothesis="Plain least squares on numeric columns already removes most of the error.",
        rationale="Cheapest informative model; isolates what numeric features alone provide, before categories or regularisation.",
        implementation="models.build_ols_numeric: NumericMedianImputer (training-fold median) + LinearRegression; Id dropped.",
        builder="houseprices.models:build_ols_numeric", builder_kwargs={}, interpret=_interp_ols, tags=["baseline", "linear"],
    ),
    dict(
        exp_id="E003", title="Ridge alpha=10, one-hot",
        hypothesis="Adding one-hot categories with L2 shrinkage beats numeric-only OLS.",
        rationale="Houses are strongly driven by Neighborhood and quality; shrinkage tames the 200+ one-hot columns.",
        implementation="models.build_ridge(alpha=10): LazyPreprocessor (neighborhood-median LotFrontage, median, RobustScaler, one-hot) + Ridge.",
        builder="houseprices.models:build_ridge", builder_kwargs={"alpha": 10.0}, interpret=_interp_ridge, tags=["baseline", "linear"],
    ),
    dict(
        exp_id="E004", title="XGBoost defaults",
        hypothesis="An untuned gradient-boosted tree model is competitive with regularised linear models on this data.",
        rationale="Trees capture interactions and non-linearity automatically; the default run shows how much tuning must still add.",
        implementation="models.build_xgboost(): XGBRegressor library defaults (seed 42, 8 threads) on one-hot unscaled inputs.",
        builder="houseprices.models:build_xgboost", builder_kwargs={}, interpret=_interp_xgb, tags=["baseline", "boosting"],
    ),
]


def main() -> None:
    rows = []
    for spec in EXPERIMENTS:
        exp = run_and_register(feature_set=FEATURE_SET, train_row_filter="keep_all", **spec)
        rows.append({"id": exp.id, "title": exp.title, "cv_mean": exp.cv_mean, "cv_std": exp.cv_std,
                     "seconds": exp.fit_seconds})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
