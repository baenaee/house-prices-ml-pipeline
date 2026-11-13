"""Scikit-learn preprocessing: everything that LEARNS from data lives here.

All estimators in this module are fitted on the training fold only. Put the result of
``make_preprocessor`` inside a ``Pipeline`` together with the model, and cross-validation
will refit it on every training fold automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

from houseprices import config


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Fill NaN in ``value_col`` with the median of its ``group_col`` group.

    ``fit`` computes the medians from the rows it is given (the training fold) and the
    global median as a fallback for groups that are unseen or entirely missing.
    ``transform`` returns a one-column float array with the filled values.
    Input must be a DataFrame containing both columns.
    """

    def __init__(self, value_col: str = "LotFrontage", group_col: str = "Neighborhood"):
        self.value_col = value_col
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y=None) -> GroupMedianImputer:
        values = X[self.value_col]
        self.group_medians_ = values.groupby(X[self.group_col]).median().dropna().to_dict()
        self.global_median_ = float(values.median())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.value_col].astype("float64")
        fill = X[self.group_col].map(self.group_medians_).astype("float64")
        fill = fill.fillna(self.global_median_)
        return values.fillna(fill).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([self.value_col])


class SkewLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply log1p to the numeric columns that are right-skewed in the TRAINING fold.

    ``fit`` learns the column list (a fitted statistic, so it must come from the training fold
    only): every column whose skewness (NaN skipped, pandas definition) is strictly greater
    than ``threshold`` AND whose minimum is >= 0. Constant columns (skew 0) fail the default
    threshold, all-NaN columns and columns with negative values are never selected. ``transform`` applies
    ``log1p`` to exactly those fitted columns of whatever rows it receives (NaN stays NaN);
    values below 0 in unseen rows are clipped to 0 first so log1p stays defined.

    Placed BEFORE the imputer in the numeric branch: the imputed value is then a statistic of
    the transformed scale. For the median strategy this is identical to imputing first
    (log1p is monotone), for the mean strategy it is the more sensible choice.
    Note that rare 0/1 indicator columns can pass the skew test; log1p of a binary column is
    a harmless monotone relabelling.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None) -> "SkewLog1pTransformer":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self.feature_names_in_ = np.array([str(c) for c in frame.columns], dtype=object)
        values = frame.astype("float64")
        skew = values.skew(skipna=True)
        minimum = values.min(skipna=True)
        selected = (skew > self.threshold) & (minimum >= 0)
        self.log_mask_ = selected.fillna(False).to_numpy(dtype=bool)
        self.skewed_columns_ = list(self.feature_names_in_[self.log_mask_])
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.array(X, dtype="float64", copy=True)
        if arr.shape[1] != len(self.log_mask_):
            raise ValueError("SkewLog1pTransformer received a different number of columns than in fit")
        cols = self.log_mask_
        arr[:, cols] = np.log1p(np.clip(arr[:, cols], 0, None))
        return arr

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


YEAR_LIKE_COLUMNS = ("YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold")


class YearRangeClipper(TransformerMixin, BaseEstimator):
    """Clip year-like columns to the largest year seen in the TRAINING fold (NaN stays NaN).

    ``fit`` learns the range of each of ``YEAR_LIKE_COLUMNS`` that is present in the frame
    (NaN ignored; an all-NaN column is left alone). ``transform`` returns a DataFrame with the
    same columns where years above the training maximum are set to that maximum. By default only
    the UPPER bound is applied (``lower=False``): a year beyond the training maximum is impossible
    for this data (e.g. GarageYrBlt 2207, test Id 2593) while an old year such as 1895 is a real
    value. ``lower=True`` also clips at the training minimum.
    The purpose is protection at prediction time: a linear model would otherwise extrapolate
    such a year ~200 years. On the development data every year lies inside the training range,
    so within cross-validation this step changes nothing. Columns routed to the categorical
    branch (``YrSold``) never reach it, because only numeric columns are given here.
    """

    def __init__(self, lower: bool = False):
        self.lower = lower

    def fit(self, X: pd.DataFrame, y=None) -> "YearRangeClipper":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self.feature_names_in_ = np.array([str(c) for c in frame.columns], dtype=object)
        self.ranges_ = {}
        for col in YEAR_LIKE_COLUMNS:
            if col in frame.columns:
                values = frame[col].astype("float64")
                if values.notna().any():
                    self.ranges_[col] = (float(values.min()), float(values.max()))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (low, high) in self.ranges_.items():
            if col in out.columns:
                out[col] = out[col].astype("float64").clip(lower=low if self.lower else None, upper=high)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


# Predictors used by the optional KNN LotFrontage imputer (lot size and house size/quality).
KNN_LOTFRONTAGE_PREDICTORS = ["LotArea", "1stFlrSF", "GrLivArea", "OverallQual", "YearBuilt"]


class KNNLotFrontageImputer(TransformerMixin, BaseEstimator):
    """Fill NaN in column 0 (LotFrontage) from its ``n_neighbors`` nearest training houses.

    Input is a DataFrame/array whose FIRST column is LotFrontage and the others are the
    predictors. ``fit`` standardises the block and fits ``KNNImputer`` on the training fold;
    ``transform`` returns the completed LotFrontage as one unscaled column.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None) -> "KNNLotFrontageImputer":
        arr = np.asarray(X, dtype="float64")
        self.scaler_ = StandardScaler().fit(arr)
        self.knn_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.scaler_.transform(arr))
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype="float64")
        filled = self.scaler_.inverse_transform(self.knn_.transform(self.scaler_.transform(arr)))
        return filled[:, [0]]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(["LotFrontage"], dtype=object)


def _as_text(X: pd.DataFrame) -> pd.DataFrame:
    """Turn every value into text but keep NaN as NaN (so codes like MSSubClass can be one-hot)."""
    return X.apply(lambda s: s.astype(object).where(s.isna(), s.astype(str)))


def split_columns(X_sample: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) judged from the dtypes of a sample frame.

    Categorical = text columns plus ``config.NUMERIC_AS_CATEGORICAL`` (integer codes).
    """
    categorical = [
        c for c in X_sample.columns
        if not pd.api.types.is_numeric_dtype(X_sample[c]) or c in config.NUMERIC_AS_CATEGORICAL
    ]
    numeric = [c for c in X_sample.columns if c not in categorical]
    return numeric, categorical


def make_preprocessor(
    X_sample: pd.DataFrame,
    numeric_strategy: str = "median",
    scale: bool = True,
    one_hot: bool = True,
    lotfrontage: str = "neighborhood_median",
    skew_log1p: bool = False,
    clip_years: bool = False,
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for the columns of ``X_sample``.

    * numeric -> SimpleImputer(numeric_strategy) [+ RobustScaler if ``scale``]
    * categorical -> SimpleImputer(constant "Unknown") + OneHotEncoder(handle_unknown="ignore")
      (``one_hot=False`` instead keeps the imputed text columns as they are; useful only for
      models that handle text themselves)
    * LotFrontage -> GroupMedianImputer by Neighborhood (``lotfrontage="neighborhood_median"``)
      or the ordinary numeric median (``"global_median"``) or a 5-nearest-neighbour fill on
      lot/house size (``"knn"``, in-fold).
    * ``skew_log1p=True`` puts a ``SkewLog1pTransformer`` before the numeric imputer. It applies to the numeric branch only,
      not to the LotFrontage branch (which keeps its own imputation and is not log-transformed).
    * ``clip_years=True`` puts a ``YearRangeClipper`` (upper bound only) first in the numeric branch (before skew, imputation
      and scaling); a no-op on the development data, a guard against impossible years at prediction time.

    ``X_sample`` is used ONLY to read column names and dtypes, never values.
    """
    if lotfrontage not in ("neighborhood_median", "global_median", "knn"):
        raise ValueError("lotfrontage must be 'neighborhood_median', 'global_median' or 'knn'")
    numeric, categorical = split_columns(X_sample)

    transformers = []
    use_group = (
        lotfrontage == "neighborhood_median"
        and "LotFrontage" in numeric
        and "Neighborhood" in X_sample.columns
    )
    if use_group:
        numeric = [c for c in numeric if c != "LotFrontage"]
        steps = [("group_median", GroupMedianImputer("LotFrontage", "Neighborhood"))]
        if scale:
            steps.append(("scale", RobustScaler()))
        transformers.append(("lotfrontage", Pipeline(steps), ["LotFrontage", "Neighborhood"]))

    if lotfrontage == "knn" and "LotFrontage" in numeric and all(c in X_sample.columns for c in KNN_LOTFRONTAGE_PREDICTORS):
        numeric = [c for c in numeric if c != "LotFrontage"]
        steps = [("knn", KNNLotFrontageImputer())]
        if scale:
            steps.append(("scale", RobustScaler()))
        transformers.append(("lotfrontage", Pipeline(steps), ["LotFrontage", *KNN_LOTFRONTAGE_PREDICTORS]))

    if numeric:
        steps = [("clip_years", YearRangeClipper())] if clip_years else []
        if skew_log1p:
            steps.append(("skew", SkewLog1pTransformer()))
        steps.append(("impute", SimpleImputer(strategy=numeric_strategy)))
        if scale:
            steps.append(("scale", RobustScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))

    if categorical:
        steps = [
            ("to_text", FunctionTransformer(_as_text, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=config.UNKNOWN_LABEL)),
        ]
        if one_hot:
            steps.append((
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=None, sparse_output=False),
            ))
        transformers.append(("categorical", Pipeline(steps), categorical))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
