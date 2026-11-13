"""Semantic cleaning of the Ames housing tables (rules P01, V01-V03, B01-B04, G01-G03, A01-A06, C01).

Every rule is a small pure function with the signature ``rule(df, raw) -> (new_df, count)``.
``df`` is the table read with pandas defaults; ``raw`` is the same file read with
``keep_default_na=False`` so that the textual markers "None" and "NA" survive (rule P01).
The two tables are matched by ``Id``.

Design principles (the same ones as the original notebook and the cleaning log):

* The SAME functions are applied to train and test. Nothing is ever fitted here.
* Recorded values are never overwritten: rules only fill missing entries or add flag columns.
* ``SalePrice`` is never read or used. It is carried along untouched.
* Rules are idempotent: running ``clean`` on its own output changes nothing.
* Estimated numeric imputation (LotFrontage, GarageYrBlt, MasVnrArea, ...) is DEFERRED to the
  training folds (see ``preprocess.py``); this module leaves those NaN in place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from houseprices import config

Rule = tuple[pd.DataFrame, int]

MASONRY_MATERIALS = ["BrkCmn", "BrkFace", "CBlock", "Stone"]
GARAGE_DETAIL_COLS = ["GarageFinish", "GarageQual", "GarageCond"]


# ---------------------------------------------------------------------------
# Loading and the P01 helper
# ---------------------------------------------------------------------------


def load_raw(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a CSV twice: with pandas defaults (``df``) and with ``keep_default_na=False`` (``raw``).

    Rule P01: the raw copy keeps the literal strings "None" and "NA" that pandas would
    otherwise turn into NaN, so later rules can tell "valid category" from "unknown".
    """
    df = pd.read_csv(path)
    raw = pd.read_csv(path, keep_default_na=False)
    return df, raw


def original_label(df: pd.DataFrame, raw: pd.DataFrame, column: str) -> pd.Series:
    """Rule P01: the raw text of ``column`` for every row of ``df``, matched by ``Id``."""
    return df["Id"].map(raw.set_index("Id")[column])


# ---------------------------------------------------------------------------
# Masonry veneer
# ---------------------------------------------------------------------------


def rule_v01(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """V01: MasVnrType missing in ``df`` but raw text is "None" -> "None" (a valid category)."""
    df = df.copy()
    mask = df["MasVnrType"].isna() & original_label(df, raw, "MasVnrType").eq("None")
    df.loc[mask, "MasVnrType"] = "None"
    return df, int(mask.sum())


def rule_v02(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """V02: MasVnrType missing and raw text is "NA" -> "Unknown". MasVnrArea is untouched."""
    df = df.copy()
    mask = df["MasVnrType"].isna() & original_label(df, raw, "MasVnrType").eq("NA")
    df.loc[mask, "MasVnrType"] = "Unknown"
    return df, int(mask.sum())


def rule_v03(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """V03: flag MasVnrConflict when type and area disagree. No value is changed.

    Conflict = (type "None" and area > 0) or (a real material and area == 0).
    """
    df = df.copy()
    none_with_area = df["MasVnrType"].eq("None") & df["MasVnrArea"].gt(0)
    material_no_area = df["MasVnrType"].isin(MASONRY_MATERIALS) & df["MasVnrArea"].eq(0)
    df["MasVnrConflict"] = none_with_area | material_no_area
    return df, int(df["MasVnrConflict"].sum())


# ---------------------------------------------------------------------------
# Basement
# ---------------------------------------------------------------------------


def rule_b01(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """B01: TotalBsmtSF == 0 and all five basement categoricals NaN -> "NoBasement"."""
    df = df.copy()
    cols = config.BASEMENT_CAT_COLS
    mask = df["TotalBsmtSF"].eq(0) & df[cols].isna().all(axis=1)
    df.loc[mask, cols] = "NoBasement"
    return df, int(mask.sum())


def rule_b02(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """B02: TotalBsmtSF > 0 and any basement categorical NaN -> fill only the NaN with "Unknown"."""
    df = df.copy()
    cols = config.BASEMENT_CAT_COLS
    mask = df["TotalBsmtSF"].gt(0) & df[cols].isna().any(axis=1)
    df.loc[mask, cols] = df.loc[mask, cols].fillna("Unknown")
    return df, int(mask.sum())


def rule_b03(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """B03: houses labelled "NoBasement" with a missing bath count -> that count becomes 0."""
    df = df.copy()
    bath_cols = ["BsmtFullBath", "BsmtHalfBath"]
    no_basement = df["TotalBsmtSF"].eq(0) & df[config.BASEMENT_CAT_COLS].eq("NoBasement").all(axis=1)
    mask = no_basement & df[bath_cols].isna().any(axis=1)
    df.loc[no_basement, bath_cols] = df.loc[no_basement, bath_cols].fillna(0)
    return df, int(mask.sum())


def basement_fully_unrecorded(df: pd.DataFrame) -> pd.Series:
    """Rows where ALL five basement categoricals AND all six basement numerics are NaN.

    This is the data-driven trigger of B04 (director decision D5). It selects test Id 2121
    and no training row, without hard-coding the Id.
    """
    cat_missing = df[config.BASEMENT_CAT_COLS].isna().all(axis=1)
    num_missing = df[config.BASEMENT_NUM_COLS].isna().all(axis=1)
    return cat_missing & num_missing


def rule_b04(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """B04: every basement field missing -> categoricals "Unknown" and BasementStatusUnknown=True.

    Numerics stay NaN (imputed in-fold later). The flag is OR-ed with any existing flag, so
    that a second pass (when the categoricals are already "Unknown") keeps it True.
    """
    df = df.copy()
    mask = basement_fully_unrecorded(df)
    previous = df["BasementStatusUnknown"] if "BasementStatusUnknown" in df.columns else False
    df["BasementStatusUnknown"] = mask | previous
    df.loc[mask, config.BASEMENT_CAT_COLS] = "Unknown"
    return df, int(mask.sum())


# ---------------------------------------------------------------------------
# Garage
# ---------------------------------------------------------------------------


def rule_g01(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """G01: GarageCars == 0, GarageArea == 0 and 4 garage categoricals NaN -> "NoGarage".

    GarageYrBlt stays numeric (NaN).
    """
    df = df.copy()
    cols = config.GARAGE_CAT_COLS
    mask = df["GarageCars"].eq(0) & df["GarageArea"].eq(0) & df[cols].isna().all(axis=1)
    df.loc[mask, cols] = "NoGarage"
    return df, int(mask.sum())


def rule_g02(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """G02: add HasGarage (GarageType present and not "NoGarage") and GarageYearUnknown.

    GarageYearUnknown = HasGarage and GarageYrBlt is NaN. The count returned is the number
    of GarageYearUnknown houses.
    """
    df = df.copy()
    df["HasGarage"] = df["GarageType"].notna() & df["GarageType"].ne("NoGarage")
    df["GarageYearUnknown"] = df["HasGarage"] & df["GarageYrBlt"].isna()
    return df, int(df["GarageYearUnknown"].sum())


def rule_g03(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """G03: HasGarage and any of GarageFinish/Qual/Cond NaN -> fill only the NaN with "Unknown"."""
    df = df.copy()
    mask = df["HasGarage"] & df[GARAGE_DETAIL_COLS].isna().any(axis=1)
    df.loc[mask, GARAGE_DETAIL_COLS] = df.loc[mask, GARAGE_DETAIL_COLS].fillna("Unknown")
    return df, int(mask.sum())


# ---------------------------------------------------------------------------
# Amenities: pool, fireplace, misc feature, alley, fence
# ---------------------------------------------------------------------------


def rule_a01(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """A01: PoolQC NaN and PoolArea == 0 -> "NoPool"."""
    df = df.copy()
    mask = df["PoolQC"].isna() & df["PoolArea"].eq(0)
    df.loc[mask, "PoolQC"] = "NoPool"
    return df, int(mask.sum())


def rule_a02(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """A02: FireplaceQu NaN and Fireplaces == 0 -> "NoFireplace"."""
    df = df.copy()
    mask = df["FireplaceQu"].isna() & df["Fireplaces"].eq(0)
    df.loc[mask, "FireplaceQu"] = "NoFireplace"
    return df, int(mask.sum())


def rule_a03(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """A03: raw MiscFeature == "NA", value NaN and MiscVal == 0 -> "NoMiscFeature"."""
    df = df.copy()
    original = original_label(df, raw, "MiscFeature")
    mask = original.eq("NA") & df["MiscFeature"].isna() & df["MiscVal"].eq(0)
    df.loc[mask, "MiscFeature"] = "NoMiscFeature"
    return df, int(mask.sum())


def rule_a04(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """A04: raw Alley == "NA" and value NaN -> "NoAlleyAccess"."""
    df = df.copy()
    mask = original_label(df, raw, "Alley").eq("NA") & df["Alley"].isna()
    df.loc[mask, "Alley"] = "NoAlleyAccess"
    return df, int(mask.sum())


def rule_a05(df: pd.DataFrame, raw: pd.DataFrame) -> Rule:
    """A05: raw Fence == "NA" and value NaN -> "NoFence"."""
    df = df.copy()
    mask = original_label(df, raw, "Fence").eq("NA") & df["Fence"].isna()
    df.loc[mask, "Fence"] = "NoFence"
    return df, int(mask.sum())


def rule_a06(df: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """A06: contradictions become "Unknown" and are flagged.

    * PoolArea > 0 and PoolQC NaN -> PoolQC "Unknown".
    * MiscVal > 0 and MiscFeature NaN -> MiscFeature "Unknown".
    * Flag MiscFeatureConflict = MiscVal > 0 and MiscFeature is NaN or "Unknown"
      (so the flag survives a second pass).
    Returns the two counts as a dict.
    """
    df = df.copy()
    pool = df["PoolArea"].gt(0) & df["PoolQC"].isna()
    misc = df["MiscVal"].gt(0) & df["MiscFeature"].isna()
    df["MiscFeatureConflict"] = df["MiscVal"].gt(0) & (
        df["MiscFeature"].isna() | df["MiscFeature"].eq("Unknown")
    )
    df.loc[pool, "PoolQC"] = "Unknown"
    df.loc[misc, "MiscFeature"] = "Unknown"
    return df, {"A06_pool": int(pool.sum()), "A06_misc": int(misc.sum())}


# ---------------------------------------------------------------------------
# C01 (reconstructed)
# ---------------------------------------------------------------------------


def rule_c01(df: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """C01: remaining NaN in ``config.C01_COLS`` -> "Unknown".

    RECONSTRUCTION: the original notebook cell defining ``c01_cols`` is lost in every
    available copy. This function re-implements the behaviour documented in the cleaning
    log (p. 10) and visible in the saved verification output (1 entry / 1 house in train,
    12 entries / 9 houses in test). It is NOT recovered code.
    """
    df = df.copy()
    cols = config.C01_COLS
    missing = df[cols].isna()
    df[cols] = df[cols].fillna("Unknown")
    return df, {"C01_entries": int(missing.sum().sum()), "C01_houses": int(missing.any(axis=1).sum())}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Notebook order (cells 49-67). Each entry: (rule id used in the counts dict, function).
SIMPLE_RULES = [
    ("V01", rule_v01), ("B01", rule_b01), ("B03", rule_b03), ("G01", rule_g01),
    ("A01", rule_a01), ("A02", rule_a02), ("A03", rule_a03), ("A04", rule_a04),
    ("A05", rule_a05), ("V02", rule_v02), ("V03", rule_v03), ("B02", rule_b02),
    ("G02", rule_g02), ("G03", rule_g03),
]


def clean(df: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply all seventeen rules in the notebook's order; return (clean_df, counts).

    Order: V01 B01 B03 G01 A01-A05 V02 V03 B02 G02 G03 A06 C01 B04.
    ``counts`` maps a rule id to the number of houses (or entries) it changed or flagged.
    Pure: the input frames are not modified.
    """
    counts: dict[str, int] = {}
    out = df.copy()
    for rule_id, rule in SIMPLE_RULES:
        out, counts[rule_id] = rule(out, raw)
    out, a06 = rule_a06(out, raw)
    counts.update(a06)
    out, c01 = rule_c01(out, raw)
    counts.update(c01)
    out, counts["B04"] = rule_b04(out, raw)
    return out, counts


def clean_train_test() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """Load and clean train and test with the same function; return (train, test, counts).

    ``counts`` is ``{"train": {...}, "test": {...}}``.
    """
    train, raw_train = load_raw(config.TRAIN_CSV)
    test, raw_test = load_raw(config.TEST_CSV)
    train_clean, train_counts = clean(train, raw_train)
    test_clean, test_counts = clean(test, raw_test)
    # A4: audit columns for both frames (adds columns only; values are never edited).
    train_clean = add_anomaly_flags(train_clean)
    test_clean = add_anomaly_flags(test_clean)
    return train_clean, test_clean, {"train": train_counts, "test": test_counts}


# ---------------------------------------------------------------------------
# Anomaly flags (task A4) - only ADD columns, never edit recorded values
# ---------------------------------------------------------------------------


def add_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add three boolean audit columns; no recorded value is changed.

    * GarageYrBltImplausible: GarageYrBlt > YrSold (e.g. test Id 2593, year 2207).
    * SoldBeforeBuilt: YrSold < YearBuilt (e.g. test Id 2550).
    * RemodBeforeBuilt: YearRemodAdd < YearBuilt (e.g. test Id 1877).
    """
    df = df.copy()
    df["GarageYrBltImplausible"] = df["GarageYrBlt"].gt(df["YrSold"])
    df["SoldBeforeBuilt"] = df["YrSold"].lt(df["YearBuilt"])
    df["RemodBeforeBuilt"] = df["YearRemodAdd"].lt(df["YearBuilt"])
    return df
