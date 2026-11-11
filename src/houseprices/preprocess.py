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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

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
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for the columns of ``X_sample``.

    * numeric -> SimpleImputer(numeric_strategy) [+ RobustScaler if ``scale``]
    * categorical -> SimpleImputer(constant "Unknown") + OneHotEncoder(handle_unknown="ignore")
      (``one_hot=False`` instead keeps the imputed text columns as they are; useful only for
      models that handle text themselves)
    * LotFrontage -> GroupMedianImputer by Neighborhood (``lotfrontage="neighborhood_median"``)
      or the ordinary numeric median (``"global_median"``).

    ``X_sample`` is used ONLY to read column names and dtypes, never values.
    """
    if lotfrontage not in ("neighborhood_median", "global_median"):
        raise ValueError("lotfrontage must be 'neighborhood_median' or 'global_median'")
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

    if numeric:
        steps = [("impute", SimpleImputer(strategy=numeric_strategy))]
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
