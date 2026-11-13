"""Hand-crafted features, organised in named groups for ablation experiments.

Every group is a PURE function: it only uses the columns of the frame it is given, row by
row. Nothing is fitted, so the functions are safe to call on train, test or a single CV fold.
Estimated fills (medians, group medians) belong to ``preprocess.py`` and happen inside folds.

Use ``build_features(df, ["base", "areas", ...])`` to assemble a feature table.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from houseprices import config

# Audit columns that describe anomalies; they are not model inputs.
ANOMALY_FLAG_COLS = ["GarageYrBltImplausible", "SoldBeforeBuilt", "RemodBeforeBuilt"]

# Columns whose quality labels become ordinal numbers in the "ordinal_quality" group.
ORDINAL_QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def total_sf(df: pd.DataFrame) -> pd.Series:
    """TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF (NaN if the basement area is NaN)."""
    return df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]


def quality_to_number(column: pd.Series) -> pd.Series:
    """Map quality labels to numbers: Po=1 ... Ex=5, absence labels=0, anything else -> NaN.

    "Unknown" deliberately becomes NaN so that it is imputed inside each training fold
    instead of receiving an arbitrary rank.
    """
    mapping: dict[str, float] = dict(config.QUALITY_SCALE)
    mapping.update({label: 0 for label in config.ABSENCE_LABELS})
    return column.map(mapping).astype("float64")


# ---------------------------------------------------------------------------
# Feature groups: each takes a cleaned frame and returns only its own columns
# ---------------------------------------------------------------------------


def group_base(df: pd.DataFrame) -> pd.DataFrame:
    """All cleaned columns except Id, SalePrice, train-constant flags and anomaly flags.

    Boolean flags are cast to 0/1 integers so every downstream imputer sees numbers.
    """
    drop = [config.ID_COL, config.TARGET, *config.CONSTANT_IN_TRAIN_FLAGS, *ANOMALY_FLAG_COLS]
    base = df.drop(columns=[c for c in drop if c in df.columns])
    bool_cols = base.select_dtypes(include="bool").columns
    return base.astype({c: "int64" for c in bool_cols})


def group_areas(df: pd.DataFrame) -> pd.DataFrame:
    """TotalSF and TotalPorchSF (sum of the five porch/deck areas)."""
    out = pd.DataFrame(index=df.index)
    out["TotalSF"] = total_sf(df)
    out["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    return out


def group_baths(df: pd.DataFrame) -> pd.DataFrame:
    """TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath."""
    out = pd.DataFrame(index=df.index)
    out["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    return out


def group_ages(df: pd.DataFrame) -> pd.DataFrame:
    """HouseAge, RemodAge and GarageAge, in years at the time of sale.

    * HouseAge  = max(YrSold - YearBuilt, 0)      (test Id 2550 was sold before it was built)
    * RemodAge  = max(YrSold - YearRemodAdd, 0)
    * GarageAge = YrSold - GarageYrBlt only when GarageYrBlt is present and <= YrSold,
      otherwise NaN (no garage, missing year, or an implausible year such as 2207).
      The NaN is imputed in-fold by preprocess.py.
    """
    out = pd.DataFrame(index=df.index)
    out["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    out["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    plausible = df["GarageYrBlt"].notna() & df["GarageYrBlt"].le(df["YrSold"])
    out["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).where(plausible)
    return out


def group_quality_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """OverallQual * GrLivArea and OverallQual * TotalSF."""
    out = pd.DataFrame(index=df.index)
    out["OverallQual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    out["OverallQual_x_TotalSF"] = df["OverallQual"] * total_sf(df)
    return out


def group_ordinal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric twin ``<col>_ord`` for each quality column; the original text column is kept."""
    out = pd.DataFrame(index=df.index)
    for col in ORDINAL_QUALITY_COLS:
        out[f"{col}_ord"] = quality_to_number(df[col])
    return out


def group_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """The three anomaly flags as 0/1 integers (GarageYrBltImplausible, SoldBeforeBuilt, RemodBeforeBuilt).

    Requires the flag columns produced by ``cleaning.clean_train_test`` / ``add_anomaly_flags``.
    Never part of "base"; tested only as an extra group in the ablation.
    """
    return df[ANOMALY_FLAG_COLS].astype("int64")


FEATURE_SETS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "base": group_base,
    "areas": group_areas,
    "baths": group_baths,
    "ages": group_ages,
    "quality_interactions": group_quality_interactions,
    "ordinal_quality": group_ordinal_quality,
    "anomaly_flags": group_anomaly_flags,
}


def build_features(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Concatenate the requested feature groups, in the order given. Pure: nothing is fitted.

    Raises ``KeyError`` for an unknown group name and ``ValueError`` if two groups would
    produce the same column name.
    """
    unknown = [g for g in groups if g not in FEATURE_SETS]
    if unknown:
        raise KeyError(f"Unknown feature groups {unknown}; choose from {sorted(FEATURE_SETS)}")
    parts = [FEATURE_SETS[g](df) for g in groups]
    out = pd.concat(parts, axis=1)
    if out.columns.duplicated().any():
        raise ValueError("Feature groups produced duplicate column names")
    return out
