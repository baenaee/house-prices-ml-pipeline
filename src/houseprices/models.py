"""Model builders: every ``build_<name>`` returns a COMPLETE scikit-learn Pipeline.

"Complete" means the pipeline accepts the cleaned DataFrame (with its ``Id`` column) and
contains the preprocessing as well as the model. Cross-validation therefore refits the
imputers, encoders and scalers on each training fold, and nothing leaks from the
validation fold.

Why the preprocessor is built lazily
------------------------------------
``preprocess.make_preprocessor`` needs a sample frame to read column names and dtypes.
A registry builder is called with keyword arguments only (no data), so :class:`LazyPreprocessor`
builds the real ``ColumnTransformer`` inside ``fit`` from the training fold it receives.
Only names and dtypes are read for that, never values, so this is equivalent to building it
up front but keeps ``build_xxx(**kwargs)`` free of any data access.

Conventions
-----------
* ``preprocessor_kwargs`` is forwarded to ``make_preprocessor`` (e.g. ``{"scale": False}``).
* ``drop_cols`` (default ``["Id"]``) are removed before preprocessing: an identifier carries
  no price information and would only add noise.
* Seeds are fixed at 42 and every boosting library is limited to ``N_THREADS`` threads.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from houseprices import config, preprocess

N_THREADS = 8          # the machine has 20 logical cores; keep headroom for other agents
SEED = config.SEED

# Tree models do not need scaled inputs; scaling would only cost time and obscure the data.
TREE_PREPROCESSOR_DEFAULTS: dict[str, Any] = {"scale": False}


# ---------------------------------------------------------------------------
# Preprocessing wrappers
# ---------------------------------------------------------------------------


class LazyPreprocessor(TransformerMixin, BaseEstimator):
    """Drop identifier columns, then apply ``make_preprocessor`` built from the fit data.

    Parameters
    ----------
    preprocessor_kwargs : forwarded to ``preprocess.make_preprocessor``.
    drop_cols : columns removed first (default ``["Id"]``); missing ones are ignored.
    scaler : "robust" keeps the shared preprocessor's RobustScaler; "standard" swaps it for
        StandardScaler (only meaningful when ``scale`` is True). Lets us compare scalers
        without editing ``preprocess.py``.
    pandas_output : return a DataFrame (needed by CatBoost to see text columns).
    """

    def __init__(
        self,
        preprocessor_kwargs: dict | None = None,
        drop_cols: list[str] | None = None,
        scaler: str = "robust",
        pandas_output: bool = False,
    ):
        self.preprocessor_kwargs = preprocessor_kwargs
        self.drop_cols = drop_cols
        self.scaler = scaler
        self.pandas_output = pandas_output

    def _drop(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = ["Id"] if self.drop_cols is None else list(self.drop_cols)
        return X.drop(columns=[c for c in cols if c in X.columns])

    def fit(self, X: pd.DataFrame, y=None) -> LazyPreprocessor:
        if self.scaler not in ("robust", "standard"):
            raise ValueError("scaler must be 'robust' or 'standard'")
        X = self._drop(X)
        ct = preprocess.make_preprocessor(X, **(self.preprocessor_kwargs or {}))
        if self.scaler == "standard":
            for name in ("numeric", "lotfrontage"):
                if f"{name}__scale" in ct.get_params():
                    ct.set_params(**{f"{name}__scale": StandardScaler()})
        if self.pandas_output:
            ct.set_output(transform="pandas")
        self.column_transformer_ = ct.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        return self.column_transformer_.transform(self._drop(X))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.column_transformer_.get_feature_names_out()


class NumericMedianImputer(TransformerMixin, BaseEstimator):
    """Keep only the numeric columns and fill gaps with the training-fold median.

    This is the deliberately naive preprocessing of the plain OLS baseline: no categories,
    no scaling. The numeric/categorical split is the shared one from ``preprocess``.
    """

    def __init__(self, drop_cols: list[str] | None = None):
        self.drop_cols = drop_cols

    def fit(self, X: pd.DataFrame, y=None) -> NumericMedianImputer:
        drop = ["Id"] if self.drop_cols is None else list(self.drop_cols)
        numeric, _ = preprocess.split_columns(X.drop(columns=[c for c in drop if c in X.columns]))
        self.columns_ = numeric
        self.imputer_ = SimpleImputer(strategy="median").fit(X[numeric])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.imputer_.transform(X[self.columns_])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_)


class CatBoostNative(RegressorMixin, BaseEstimator):
    """CatBoostRegressor that treats every text column of its input as a native categorical.

    CatBoost encodes categories itself (ordered target statistics, computed in a leakage-safe
    way inside each training set), so it should receive the imputed TEXT columns rather than
    one-hot columns. The wrapper looks at the dtypes it is fit on and passes the object
    columns as ``cat_features``.
    """

    def __init__(self, params: dict | None = None):
        self.params = params

    def _make(self):
        from catboost import CatBoostRegressor

        defaults = dict(random_seed=SEED, thread_count=N_THREADS, verbose=0, allow_writing_files=False)
        return CatBoostRegressor(**{**defaults, **(self.params or {})})

    def fit(self, X: pd.DataFrame, y) -> CatBoostNative:
        cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("category", "string")]
        self.cat_cols_ = cat_cols
        self.model_ = self._make().fit(X, np.asarray(y), cat_features=cat_cols)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X), dtype=float)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _pipeline(model, preprocessor_kwargs, drop_cols, scaler="robust", pandas_output=False, tree=False) -> Pipeline:
    """Shared recipe: LazyPreprocessor + model."""
    kwargs = {**(TREE_PREPROCESSOR_DEFAULTS if tree else {}), **(preprocessor_kwargs or {})}
    pre = LazyPreprocessor(kwargs, drop_cols, scaler, pandas_output)
    return Pipeline([("prep", pre), ("model", model)])


def build_mean_baseline(**_ignored) -> Pipeline:
    """Predict the training mean of log-price for everybody: the score any model must beat."""
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def build_ols_numeric(drop_cols: list[str] | None = None) -> Pipeline:
    """Ordinary least squares on numeric columns only (median-imputed), no regularisation."""
    return Pipeline([("prep", NumericMedianImputer(drop_cols)), ("model", LinearRegression())])


def build_ridge(alpha: float = 10.0, preprocessor_kwargs: dict | None = None,
                drop_cols: list[str] | None = None, scaler: str = "robust") -> Pipeline:
    """L2-penalised linear model on scaled numeric + one-hot categorical inputs."""
    return _pipeline(Ridge(alpha=alpha, random_state=SEED), preprocessor_kwargs, drop_cols, scaler)


def build_lasso(alpha: float = 0.001, max_iter: int = 20000, preprocessor_kwargs: dict | None = None,
                drop_cols: list[str] | None = None, scaler: str = "robust") -> Pipeline:
    """L1-penalised linear model (sparse); needs many iterations to converge on one-hot data."""
    return _pipeline(Lasso(alpha=alpha, max_iter=max_iter, random_state=SEED),
                     preprocessor_kwargs, drop_cols, scaler)


def build_elasticnet(alpha: float = 0.001, l1_ratio: float = 0.5, max_iter: int = 20000,
                     preprocessor_kwargs: dict | None = None, drop_cols: list[str] | None = None,
                     scaler: str = "robust") -> Pipeline:
    """Mix of L1 and L2 penalties; ``l1_ratio`` 1.0 is Lasso, 0.0 is (almost) Ridge."""
    return _pipeline(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=SEED),
                     preprocessor_kwargs, drop_cols, scaler)


def build_xgboost(preprocessor_kwargs: dict | None = None, drop_cols: list[str] | None = None,
                  **params) -> Pipeline:
    """XGBoost regressor on one-hot inputs. ``params`` override the defaults (library defaults
    except seed and threads)."""
    from xgboost import XGBRegressor

    defaults = dict(random_state=SEED, n_jobs=N_THREADS)
    model = XGBRegressor(**{**defaults, **params})
    return _pipeline(model, preprocessor_kwargs, drop_cols, tree=True)


def build_lightgbm(preprocessor_kwargs: dict | None = None, drop_cols: list[str] | None = None,
                   **params) -> Pipeline:
    """LightGBM regressor on one-hot inputs (quiet, seeded, thread-limited)."""
    from lightgbm import LGBMRegressor

    defaults = dict(random_state=SEED, n_jobs=N_THREADS, verbose=-1)
    model = LGBMRegressor(**{**defaults, **params})
    return _pipeline(model, preprocessor_kwargs, drop_cols, tree=True)


def build_catboost(preprocessor_kwargs: dict | None = None, drop_cols: list[str] | None = None,
                   native_categoricals: bool = False, **params) -> Pipeline:
    """CatBoost regressor.

    ``native_categoricals=False``: one-hot through the shared preprocessor.
    ``native_categoricals=True``: the preprocessor only imputes text columns and CatBoost
    encodes them itself (its main selling point on categorical-heavy data like Ames).
    """
    if native_categoricals:
        pre_kwargs = {**TREE_PREPROCESSOR_DEFAULTS, **(preprocessor_kwargs or {}), "one_hot": False}
        pre = LazyPreprocessor(pre_kwargs, drop_cols, pandas_output=True)
        return Pipeline([("prep", pre), ("model", CatBoostNative(params))])
    from catboost import CatBoostRegressor

    defaults = dict(random_seed=SEED, thread_count=N_THREADS, verbose=0, allow_writing_files=False)
    model = CatBoostRegressor(**{**defaults, **params})
    return _pipeline(model, preprocessor_kwargs, drop_cols, tree=True)


# ---------------------------------------------------------------------------
# Optuna search spaces (bounded, so a handful of trials is meaningful)
# ---------------------------------------------------------------------------


def _xgboost_space(trial) -> dict:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 2000, step=100),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
    )


def _lightgbm_space(trial) -> dict:
    # LightGBM's min_child_weight is a hessian threshold that is nearly inert for squared
    # error; min_child_samples (rows per leaf) is the knob that actually controls overfitting.
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 2000, step=100),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,   # LightGBM ignores `subsample` unless bagging frequency is > 0
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
    )


def _catboost_space(trial) -> dict:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        iterations=trial.suggest_int("iterations", 300, 2000, step=100),
        depth=trial.suggest_int("depth", 3, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        rsm=trial.suggest_float("rsm", 0.5, 1.0),   # CatBoost's name for colsample
    )


OPTUNA_SPACES: dict[str, Callable] = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "catboost": _catboost_space,
}

BUILDERS: dict[str, Callable[..., Pipeline]] = {
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
}
