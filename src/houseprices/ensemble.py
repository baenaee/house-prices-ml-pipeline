"""Ensembles that are safe to evaluate: blending and NESTED stacking.

Why this module exists
----------------------
A stack has two layers: base models, and a meta-learner that combines their predictions.
The meta-learner is itself a model with parameters, so it must be scored on rows it was
never fit on, exactly like any other model.

The classic mistake: fit the base models, collect their out-of-fold (OOF) predictions for
the whole training set, fit the meta-learner on them, and then report the meta-learner's
CV score on those SAME rows. The meta-learner has already seen those rows and their
targets, so the score is optimistic. With flexible meta-learners or many base models the
optimism can be large.

The fix is NESTED cross-validation. The outer loop (``validation.cross_validate``) holds
out a validation fold. Everything the stack learns, including the meta-learner's
training data, is produced INSIDE the outer-train rows by an inner K-fold. The outer
validation fold is touched once, only for scoring. We implement the inner loop inside
``fit()`` of the estimator, so the outer protocol stays the single source of truth and
any ensemble reproduces from the registry with ``python -m houseprices.evaluate``.

Fitting per outer fold, for ``NestedStackingRegressor.fit(X_tr, y_tr)``:

1. inner K-fold on (X_tr, y_tr): each base model is fit on inner-train and predicts the
   inner-validation rows -> a matrix of OOF predictions covering every row of X_tr;
2. the meta-learner is fit on that matrix (target y_tr);
3. every base model is refit on ALL of X_tr (they will predict unseen rows later).

``predict(X_val)`` sends X_val through the refit base models and then the meta-learner.

Training-row filters (outlier policies) are applied by ``validation.iter_folds`` to the
outer-train rows before ``fit`` is called, so rows removed from training never enter the
inner CV either. Outer validation rows are never filtered.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from houseprices import config, data, validation

NamedEstimator = tuple[str, BaseEstimator]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _take(X, positions: np.ndarray):
    """Row selection that works for DataFrames/Series (iloc) and numpy arrays."""
    return X.iloc[positions] if hasattr(X, "iloc") else np.asarray(X)[positions]


def _fit_predict(estimator: BaseEstimator, X_tr, y_tr, X_va) -> np.ndarray:
    """Fit a fresh clone on the training part and predict the validation part."""
    model = clone(estimator)
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_va), dtype=float)


def inner_oof_matrix(
    base_estimators: Sequence[NamedEstimator], X, y, n_splits: int, seed: int
) -> np.ndarray:
    """Out-of-fold predictions of every base model, shape (n_rows, n_models).

    Row ``i`` of column ``j`` was predicted by model ``j`` fit on folds that do NOT
    contain row ``i``. This is the only kind of prediction a meta-learner may train on.
    """
    y_arr = np.asarray(y, dtype=float)
    oof = np.full((len(y_arr), len(base_estimators)), np.nan)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in splitter.split(y_arr):
        for j, (_, est) in enumerate(base_estimators):
            oof[va, j] = _fit_predict(est, _take(X, tr), y_arr[tr], _take(X, va))
    return oof


def _fit_all(base_estimators: Sequence[NamedEstimator], X, y) -> list[NamedEstimator]:
    """Refit clones of every base model on all rows received."""
    return [(name, clone(est).fit(X, y)) for name, est in base_estimators]


def _predict_matrix(fitted: Sequence[NamedEstimator], X) -> np.ndarray:
    """Column-stack the predictions of already-fitted base models."""
    return np.column_stack([np.asarray(est.predict(X), dtype=float) for _, est in fitted])


# ---------------------------------------------------------------------------
# Blend weights
# ---------------------------------------------------------------------------


def fit_blend_weights(
    oof_matrix: np.ndarray,
    y_log: np.ndarray,
    non_negative: bool = True,
    sum_to_one: bool = True,
) -> np.ndarray:
    """Least-squares weights ``w`` minimising ``||oof_matrix @ w - y_log||^2``.

    The constraints are regularisers: non-negative weights stop the blend from
    "subtracting" a model (which exploits noise), and weights summing to one keep the
    blend on the scale of the target. Call this ONLY with out-of-fold predictions.
    """
    A = np.asarray(oof_matrix, dtype=float)
    y = np.asarray(y_log, dtype=float)
    k = A.shape[1]
    if non_negative and not sum_to_one:
        return nnls(A, y)[0]
    if not non_negative and not sum_to_one:
        return np.linalg.lstsq(A, y, rcond=None)[0]

    def loss(w: np.ndarray) -> float:
        return float(np.sum((A @ w - y) ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * A.T @ (A @ w - y)

    res = minimize(
        loss,
        x0=np.full(k, 1.0 / k),
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, None)] * k if non_negative else None,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = res.x
    if non_negative:  # remove tiny negative numerical noise, then re-normalise
        w = np.clip(w, 0.0, None)
        w = w / w.sum()
    return w


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


class BlendingRegressor(BaseEstimator, RegressorMixin):
    """Weighted average of base models.

    ``weights`` given: fixed weights (no learning, so no leakage possible).
    ``weights=None``: weights are learned inside ``fit`` from INNER out-of-fold
    predictions of the rows received, so the outer validation fold never influences them.
    """

    def __init__(
        self,
        base_estimators: list[NamedEstimator],
        weights: list[float] | None = None,
        inner_cv_splits: int = config.CV_N_SPLITS,
        seed: int = config.SEED,
        non_negative: bool = True,
        sum_to_one: bool = True,
    ):
        self.base_estimators = base_estimators
        self.weights = weights
        self.inner_cv_splits = inner_cv_splits
        self.seed = seed
        self.non_negative = non_negative
        self.sum_to_one = sum_to_one

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        if self.weights is None:
            oof = inner_oof_matrix(self.base_estimators, X, y_arr, self.inner_cv_splits, self.seed)
            self.weights_ = fit_blend_weights(oof, y_arr, self.non_negative, self.sum_to_one)
        else:
            if len(self.weights) != len(self.base_estimators):
                raise ValueError("need one weight per base estimator")
            self.weights_ = np.asarray(self.weights, dtype=float)
        self.fitted_bases_ = _fit_all(self.base_estimators, X, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        return _predict_matrix(self.fitted_bases_, X) @ self.weights_


class NestedStackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking whose meta-learner only ever trains on inner out-of-fold predictions.

    See the module docstring. ``passthrough=True`` also gives the meta-learner the raw
    features, which requires ``X`` to be purely numeric; with the project's mixed-type
    raw frames put a preprocessing step inside the meta estimator instead.
    """

    def __init__(
        self,
        base_estimators: list[NamedEstimator],
        meta_estimator: BaseEstimator,
        inner_cv_splits: int = config.CV_N_SPLITS,
        seed: int = config.SEED,
        passthrough: bool = False,
    ):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.inner_cv_splits = inner_cv_splits
        self.seed = seed
        self.passthrough = passthrough

    def _meta_features(self, base_matrix: np.ndarray, X) -> np.ndarray:
        if not self.passthrough:
            return base_matrix
        return np.hstack([base_matrix, np.asarray(X, dtype=float)])

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        # (1) inner OOF predictions, computed only from the rows we were given
        oof = inner_oof_matrix(self.base_estimators, X, y_arr, self.inner_cv_splits, self.seed)
        # (2) meta-learner trained on those out-of-sample predictions
        self.meta_ = clone(self.meta_estimator).fit(self._meta_features(oof, X), y_arr)
        # (3) bases refit on all received rows, used to predict unseen rows later
        self.fitted_bases_ = _fit_all(self.base_estimators, X, y_arr)
        self.inner_oof_ = oof
        return self

    def predict(self, X) -> np.ndarray:
        base = _predict_matrix(self.fitted_bases_, X)
        return np.asarray(self.meta_.predict(self._meta_features(base, X)), dtype=float)


# ---------------------------------------------------------------------------
# Labelled optimistic diagnostics (never use these as evidence of performance)
# ---------------------------------------------------------------------------


def optimistic_oof_stack_score(
    base_estimators: Sequence[NamedEstimator], meta_estimator: BaseEstimator, X, y_log, cv=None
) -> float:
    """OPTIMISTIC: fit the meta-learner on all OOF rows and score it on those same rows."""
    cv = cv if cv is not None else validation.make_cv()
    y = np.asarray(y_log, dtype=float)
    oof = np.full((len(y), len(base_estimators)), np.nan)
    for tr, va in cv.split(y):
        for j, (_, est) in enumerate(base_estimators):
            oof[va, j] = _fit_predict(est, _take(X, tr), y[tr], _take(X, va))
    meta = clone(meta_estimator).fit(oof, y)
    return validation.rmse_log(y, meta.predict(oof))


def in_sample_stack_score(
    base_estimators: Sequence[NamedEstimator], meta_estimator: BaseEstimator, X, y_log
) -> float:
    """VERY OPTIMISTIC: bases and meta both fit and scored on the same rows (the leak
    nested stacking exists to prevent)."""
    y = np.asarray(y_log, dtype=float)
    fitted = _fit_all(base_estimators, X, y)
    feats = _predict_matrix(fitted, X)
    meta = clone(meta_estimator).fit(feats, y)
    return validation.rmse_log(y, meta.predict(feats))


# ---------------------------------------------------------------------------
# Diversity analysis
# ---------------------------------------------------------------------------


def prediction_correlation(oof_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of OOF predictions (columns = models)."""
    return pd.DataFrame(oof_matrix).corr()


def residual_correlation(oof_matrix: pd.DataFrame, y_log: np.ndarray | pd.Series) -> pd.DataFrame:
    """Pairwise correlation of residuals. Low values mean errors differ, so blending helps."""
    oof = pd.DataFrame(oof_matrix)
    resid = oof.sub(np.asarray(y_log, dtype=float), axis=0)
    return resid.corr()


def diversity_report(oof_matrix: pd.DataFrame, y_log: np.ndarray | pd.Series) -> pd.DataFrame:
    """One row per model: RMSE and its mean residual correlation with the other models."""
    oof = pd.DataFrame(oof_matrix)
    rc = residual_correlation(oof, y_log).to_numpy()
    k = len(oof.columns)
    mean_other = [(rc[i].sum() - 1.0) / (k - 1) if k > 1 else np.nan for i in range(k)]
    rmse = [validation.rmse_log(np.asarray(y_log, dtype=float), oof.iloc[:, i]) for i in range(k)]
    return pd.DataFrame(
        {"rmse": rmse, "mean_residual_corr_with_others": mean_other}, index=oof.columns
    )


# ---------------------------------------------------------------------------
# Registry builders (reproducible via python -m houseprices.evaluate)
# ---------------------------------------------------------------------------


def _build_bases(base_builders: list[str], base_kwargs: list[dict] | None) -> list[NamedEstimator]:
    kwargs = base_kwargs if base_kwargs is not None else [{} for _ in base_builders]
    if len(kwargs) != len(base_builders):
        raise ValueError("base_kwargs must have one dict per base builder")
    out: list[NamedEstimator] = []
    for i, (path, kw) in enumerate(zip(base_builders, kwargs)):
        out.append((f"{path.split(':')[-1]}_{i}", data.resolve_callable(path)(**kw)))
    return out


def build_blend(
    base_builders: list[str],
    base_kwargs: list[dict] | None = None,
    weights: list[float] | None = None,
) -> BlendingRegressor:
    """Blend of registry-referenced base pipelines; ``weights=None`` learns them (nested)."""
    return BlendingRegressor(_build_bases(base_builders, base_kwargs), weights=weights)


def build_stack(
    base_builders: list[str],
    base_kwargs: list[dict] | None = None,
    meta: str = "ridge",
    meta_alpha: float = 1.0,
    inner_cv_splits: int = config.CV_N_SPLITS,
    passthrough: bool = False,
) -> NestedStackingRegressor:
    """Nested stack with a Ridge meta-learner (the only meta supported, on purpose:
    a low-capacity meta-learner is what keeps stacking from overfitting)."""
    if meta != "ridge":
        raise ValueError("only meta='ridge' is supported")
    return NestedStackingRegressor(
        _build_bases(base_builders, base_kwargs),
        Ridge(alpha=meta_alpha),
        inner_cv_splits=inner_cv_splits,
        passthrough=passthrough,
    )

