"""Validation protocol: target transform, metric, folds, holdout and the CV runner.

Design rules (binding, see docs/AUDIT_2026-11-11.md and the approved plan):

1. Models are fit on ``y = log1p(SalePrice)``. The metric is RMSE on that scale, which is
   exactly the competition's RMSLE.
2. Everything that learns from data (imputers, encoders, scalers, models) lives inside a
   scikit-learn ``Pipeline`` that is fit on the training part of each fold only.
3. Training-row policies such as outlier removal are applied AFTER the split and ONLY to
   the training part of a fold. The validation fold is always the full, unchanged fold.
4. The 15 % holdout is carved once, stored in ``experiments/holdout_ids.json`` and used a
   single time after the modelling procedure is frozen.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedShuffleSplit

from houseprices import config

# A training-row filter receives the training-fold features and log-target and returns a
# boolean mask of rows to KEEP. It must never see the validation fold.
TrainRowFilter = Callable[[pd.DataFrame, pd.Series], np.ndarray]


# ---------------------------------------------------------------------------
# Target transform and metric
# ---------------------------------------------------------------------------


def to_log_target(sale_price: pd.Series) -> pd.Series:
    """log1p(SalePrice); the inverse is :func:`from_log_target`."""
    return np.log1p(sale_price.astype(float)).rename(config.LOG_TARGET)


def from_log_target(y_log: np.ndarray | pd.Series) -> np.ndarray:
    """expm1 back to dollars."""
    return np.expm1(np.asarray(y_log, dtype=float))


def rmse_log(y_log_true: np.ndarray | pd.Series, y_log_pred: np.ndarray | pd.Series) -> float:
    """RMSE between two vectors already on the log1p scale (= Kaggle RMSLE)."""
    a = np.asarray(y_log_true, dtype=float)
    b = np.asarray(y_log_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmsle_dollars(price_true: np.ndarray, price_pred: np.ndarray) -> float:
    """Competition metric computed from dollar values (for cross-checking rmse_log)."""
    return rmse_log(np.log1p(price_true), np.log1p(price_pred))


# ---------------------------------------------------------------------------
# Folds and holdout
# ---------------------------------------------------------------------------


def make_cv(n_splits: int = config.CV_N_SPLITS, n_repeats: int = 1, seed: int = config.SEED):
    """Shuffled KFold (n_repeats=1) or RepeatedKFold, always seeded."""
    if n_repeats == 1:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def make_holdout_ids(
    train: pd.DataFrame,
    fraction: float = config.HOLDOUT_FRACTION,
    seed: int = config.SEED,
) -> list[int]:
    """Deterministically choose holdout house Ids, stratified by SalePrice decile."""
    deciles = pd.qcut(train[config.TARGET], q=10, labels=False, duplicates="drop")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    _, holdout_pos = next(splitter.split(train, deciles))
    return sorted(int(i) for i in train[config.ID_COL].iloc[holdout_pos])


def load_or_create_holdout_ids(train: pd.DataFrame) -> list[int]:
    """Read experiments/holdout_ids.json, creating it on first use so it is fixed forever."""
    path = config.HOLDOUT_IDS_PATH
    if path.exists():
        ids = json.loads(path.read_text(encoding="utf-8"))["holdout_ids"]
        known = set(train[config.ID_COL].tolist())
        missing = [i for i in ids if i not in known]
        if missing or len(ids) != len(set(ids)):
            raise ValueError(
                f"{path} is inconsistent with the training data (unknown or duplicate ids: {missing[:5]})"
            )
        return ids
    ids = make_holdout_ids(train)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fraction": config.HOLDOUT_FRACTION,
                "seed": config.SEED,
                "stratified_by": "SalePrice decile",
                "n_holdout": len(ids),
                "holdout_ids": ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ids


def split_development_holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (development, holdout) frames. All model selection happens on development."""
    ids = set(load_or_create_holdout_ids(train))
    mask = train[config.ID_COL].isin(ids)
    return train.loc[~mask].reset_index(drop=True), train.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    fold_scores: list[float]
    oof_pred: np.ndarray                      # log-scale OOF predictions (NaN if repeated CV)
    fit_seconds: float
    n_train_rows_per_fold: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores, ddof=1)) if len(self.fold_scores) > 1 else 0.0

    def summary(self) -> dict:
        return {
            "cv_mean": round(self.mean, 5),
            "cv_std": round(self.std, 5),
            "fold_scores": [round(s, 5) for s in self.fold_scores],
            "fit_seconds": round(self.fit_seconds, 1),
        }


def iter_folds(
    X: pd.DataFrame, y_log: pd.Series, cv, train_row_filter: TrainRowFilter | None = None
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold_number, train_positions, valid_positions).

    The optional filter shrinks ONLY the training positions. Validation positions are the
    untouched fold produced by the splitter.
    """
    for fold, (tr, va) in enumerate(cv.split(X)):
        if train_row_filter is not None:
            keep = np.asarray(train_row_filter(X.iloc[tr], y_log.iloc[tr]), dtype=bool)
            if keep.shape != tr.shape:
                raise ValueError("train_row_filter must return one boolean per training row")
            tr = tr[keep]
        yield fold, tr, va


def cross_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y_log: pd.Series,
    cv=None,
    train_row_filter: TrainRowFilter | None = None,
    fit_params: dict | None = None,
) -> CVResult:
    """Fit a fresh clone of ``estimator`` per fold and score RMSE on the log target.

    ``estimator`` must be a complete pipeline that accepts the raw (cleaned) DataFrame, so
    that all preprocessing is fit per fold.
    """
    cv = cv if cv is not None else make_cv()
    fit_params = fit_params or {}
    # OOF predictions are only well defined when every row is validated exactly once.
    # RepeatedKFold exposes `n_repeats`; plain KFold does not (treated as 1 repeat).
    is_single_pass = getattr(cv, "n_repeats", 1) == 1

    oof = np.full(len(X), np.nan)
    scores: list[float] = []
    n_rows: list[int] = []
    t0 = time.perf_counter()
    for fold, tr, va in iter_folds(X, y_log, cv, train_row_filter):
        if len(tr) == 0:
            raise ValueError(f"fold {fold}: the train_row_filter removed every training row")
        model = clone(estimator)
        model.fit(X.iloc[tr], y_log.iloc[tr], **fit_params)
        pred = np.asarray(model.predict(X.iloc[va]), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError(
                f"fold {fold}: estimator produced {int((~np.isfinite(pred)).sum())} non-finite predictions"
            )
        scores.append(rmse_log(y_log.iloc[va], pred))
        n_rows.append(len(tr))
        if is_single_pass:
            oof[va] = pred
    return CVResult(scores, oof, time.perf_counter() - t0, n_rows)


# ---------------------------------------------------------------------------
# Training-row policies (outliers). They see training rows only.
# ---------------------------------------------------------------------------


def keep_all(X: pd.DataFrame, y_log: pd.Series) -> np.ndarray:
    return np.ones(len(X), dtype=bool)


def drop_candidate_outliers(X: pd.DataFrame, y_log: pd.Series) -> np.ndarray:
    """Drop the two houses flagged by the original EDA (Ids 524 and 1299) from TRAINING only."""
    return ~X[config.ID_COL].isin(config.CANDIDATE_OUTLIER_IDS).to_numpy()


def drop_large_living_area(threshold: float = 4000.0) -> TrainRowFilter:
    """Factory: drop training houses with GrLivArea above ``threshold`` (default 4000 sq ft)."""

    def _filter(X: pd.DataFrame, y_log: pd.Series) -> np.ndarray:
        return (X["GrLivArea"] <= threshold).to_numpy()

    _filter.__name__ = f"drop_GrLivArea_gt_{int(threshold)}"
    return _filter


# ---------------------------------------------------------------------------
# Single-use holdout evaluation (binding correction 3)
# ---------------------------------------------------------------------------

HOLDOUT_USAGE_LOG = config.EXPERIMENTS_DIR / "holdout_usage.log"


class HoldoutAlreadyUsedError(RuntimeError):
    """Raised when the holdout has already been scored once."""


def holdout_usage() -> list[dict]:
    """Every recorded holdout evaluation (normally zero or one entries)."""
    if not HOLDOUT_USAGE_LOG.exists():
        return []
    return [json.loads(line) for line in HOLDOUT_USAGE_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_holdout_once(
    fitted_estimator: BaseEstimator,
    X_holdout: pd.DataFrame,
    y_holdout_log: pd.Series,
    frozen_experiment_id: str,
    note: str = "",
    force: bool = False,
) -> dict:
    """Score the single-use holdout and append the event to ``experiments/holdout_usage.log``.

    The estimator must already be fitted on the development rows (post training-row policy).
    A second call raises :class:`HoldoutAlreadyUsedError` unless ``force=True``; a forced call
    is logged with ``"forced": true`` so the report must declare that the holdout is no longer
    an unbiased estimate (binding correction 3).
    """
    used = holdout_usage()
    if used and not force:
        raise HoldoutAlreadyUsedError(
            f"holdout already scored on {used[0]['timestamp']} for {used[0]['frozen_experiment_id']}; "
            "a second evaluation would bias the estimate (pass force=True only with a documented reason)"
        )
    pred = np.asarray(fitted_estimator.predict(X_holdout), dtype=float)
    if not np.all(np.isfinite(pred)):
        raise ValueError("holdout predictions are not all finite")
    score = rmse_log(y_holdout_log, pred)
    event = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen_experiment_id": frozen_experiment_id,
        "n_holdout": int(len(X_holdout)),
        "rmse_log": round(score, 6),
        "forced": bool(force),
        "n_previous_uses": len(used),
        "note": note,
    }
    HOLDOUT_USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HOLDOUT_USAGE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    return event
