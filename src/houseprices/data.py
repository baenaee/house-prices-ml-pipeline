"""One shared entry point for "give me the data an experiment should see".

Every experiment, the evaluate command and the final training script obtain their
inputs through :func:`load_development_data`, so the cleaning, the holdout split and the
feature construction are guaranteed to be identical everywhere.

Pipeline:  raw CSV  ->  cleaning.clean_train_test()  ->  holdout split  ->  features.build_features()

Nothing in this module fits anything. Imputation, encoding and scaling happen inside the
estimator pipelines, per fold.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from houseprices import config, validation


@dataclass
class DevelopmentData:
    X: pd.DataFrame          # development rows, engineered features, includes Id
    y_log: pd.Series         # log1p(SalePrice) aligned with X
    X_holdout: pd.DataFrame  # single-use holdout rows (do not touch during model selection)
    y_holdout_log: pd.Series
    X_test: pd.DataFrame     # competition test rows, same columns as X
    feature_groups: list[str]


def parse_feature_set(feature_set: str) -> list[str]:
    """'base+areas+baths' -> ['base', 'areas', 'baths']; 'raw_cleaned' -> ['base']."""
    if feature_set in ("", "raw_cleaned", "base"):
        return ["base"]
    return [g.strip() for g in feature_set.split("+") if g.strip()]


def load_development_data(feature_set: str = "base", with_holdout_split: bool = True) -> DevelopmentData:
    """Clean, split off the holdout (once, deterministically) and build the named features."""
    from houseprices import cleaning, features  # imported lazily: written by Agent A

    train_clean, test_clean, _counts = cleaning.clean_train_test()
    if with_holdout_split:
        dev, hold = validation.split_development_holdout(train_clean)
    else:
        dev, hold = train_clean, train_clean.iloc[0:0]

    groups = parse_feature_set(feature_set)
    X = _with_id(dev, features.build_features(dev, groups))
    X_hold = _with_id(hold, features.build_features(hold, groups))
    X_test = _with_id(test_clean, features.build_features(test_clean, groups))

    y_log = validation.to_log_target(dev[config.TARGET])
    y_hold = validation.to_log_target(hold[config.TARGET]) if len(hold) else pd.Series(dtype=float)

    for frame in (X, X_hold, X_test):
        if config.TARGET in frame.columns:
            frame.drop(columns=[config.TARGET], inplace=True)
    return DevelopmentData(X, y_log, X_hold, y_hold, X_test, groups)


def _with_id(source: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Re-attach ``Id`` as the first column of a feature table.

    ``features.group_base`` deliberately excludes the identifier from the model inputs, but the
    training-row policies in ``validation`` (``drop_candidate_outliers``) select rows by Id, and
    ``predict.py`` needs it to write the submission. Every model builder drops ``Id`` before
    preprocessing, so it never reaches an estimator. (Director fix, 2026-11-13.)
    """
    if config.ID_COL in X.columns:
        return X
    X = X.copy()
    X.insert(0, config.ID_COL, source[config.ID_COL].to_numpy())
    return X


def resolve_callable(dotted: str) -> Callable:
    """'package.module:function' -> the function object."""
    module_name, _, func_name = dotted.partition(":")
    if not func_name:
        raise ValueError(f"expected 'module:function', got {dotted!r}")
    return getattr(importlib.import_module(module_name), func_name)


def get_train_row_filter(name: str) -> validation.TrainRowFilter:
    """Map a registry string to a training-row filter.

    Accepted: 'keep_all', 'drop_candidate_outliers', 'drop_GrLivArea_gt_<N>', or 'module:function'.
    """
    if name in ("", "keep_all", None):
        return validation.keep_all
    if name == "drop_candidate_outliers":
        return validation.drop_candidate_outliers
    if name.startswith("drop_GrLivArea_gt_"):
        return validation.drop_large_living_area(float(name.rsplit("_", 1)[1]))
    if ":" in name:
        return resolve_callable(name)
    raise ValueError(f"unknown train_row_filter {name!r}")


def describe(data: DevelopmentData) -> dict:
    return {
        "n_dev": int(len(data.X)),
        "n_holdout": int(len(data.X_holdout)),
        "n_test": int(len(data.X_test)),
        "n_features": int(data.X.shape[1] - 1),  # minus Id
        "feature_groups": data.feature_groups,
        "y_log_mean": float(np.mean(data.y_log)),
    }
