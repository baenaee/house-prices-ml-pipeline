"""Project-wide configuration: paths, seeds, cross-validation settings and column groups.

Everything that several modules must agree on lives here, so that a change (for
example the random seed) is made in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root by walking upwards until `data_description.txt` is found.

    Works from the package directory, from `notebooks/`, from `tests/` and from any
    working directory inside the project, so notebooks and scripts never need
    hard-coded absolute paths.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data_description.txt").exists() and (candidate / "train.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root (a folder containing data_description.txt and train.csv). "
        "The Kaggle data are not in the repository: run `python scripts/download_data.py` first."
    )


PROJECT_ROOT = find_project_root()

# The original Kaggle files stay at the project root, untouched.
TRAIN_CSV = PROJECT_ROOT / "train.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"
SAMPLE_SUBMISSION_CSV = PROJECT_ROOT / "sample_submission.csv"
DATA_DESCRIPTION_TXT = PROJECT_ROOT / "data_description.txt"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"
HOLDOUT_IDS_PATH = EXPERIMENTS_DIR / "holdout_ids.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
MODELS_DIR = OUTPUTS_DIR / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Reproducibility and validation protocol
# ---------------------------------------------------------------------------

SEED = 42

ID_COL = "Id"
TARGET = "SalePrice"
# All models are trained on log1p(SalePrice); the competition metric is RMSE on that scale.
LOG_TARGET = "log1p_SalePrice"

CV_N_SPLITS = 5            # model-selection protocol: 5-fold, shuffled, seed 42
CV_N_REPEATS_FINALISTS = 3  # finalists are re-checked with RepeatedKFold(5 x 3)
# Finalists use a DIFFERENT seed from the selection folds (audit F20, decision D17): with the
# selection seed, the first repeat of RepeatedKFold reproduces the very folds every
# configuration was chosen on and inherits their selection optimism (~+0.002 measured).
CV_SEED_FINALISTS = 7
HOLDOUT_FRACTION = 0.15     # single-use holdout, stratified by price decile, carved once

# ---------------------------------------------------------------------------
# Column groups (names exactly as they appear in the CSV files)
# ---------------------------------------------------------------------------

# Five basement descriptors and four garage descriptors treated by the cleaning rules.
BASEMENT_CAT_COLS = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
BASEMENT_NUM_COLS = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
GARAGE_CAT_COLS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
GARAGE_NUM_COLS = ["GarageCars", "GarageArea"]

# Rule C01 (reconstructed from the cleaning log, p. 8 and p. 10): categorical columns that
# were complete in train but had gaps in test, plus the single missing Electrical in train.
C01_COLS = [
    "Electrical", "MSZoning", "Utilities", "Functional",
    "Exterior1st", "Exterior2nd", "KitchenQual", "SaleType",
]

# Audit/flag columns created by the cleaning rules. They document data-quality findings.
AUDIT_FLAG_COLS = [
    "MasVnrConflict",        # V03: veneer type/area disagree (7 train, 3 test)
    "HasGarage",             # G02: garage present
    "GarageYearUnknown",     # G02: garage present but year missing (0 train, 2 test)
    "MiscFeatureConflict",   # A06: MiscVal > 0 but feature unknown (0 train, 1 test)
    "BasementStatusUnknown", # B04: every basement field missing (0 train, 1 test)
]
# Constant in the training data (always False), therefore excluded from model features.
CONSTANT_IN_TRAIN_FLAGS = ["GarageYearUnknown", "MiscFeatureConflict", "BasementStatusUnknown"]

# Numeric columns deliberately left missing by the semantic cleaning (rules N01/N02).
# They are imputed INSIDE each training fold by preprocess.py, never globally.
DEFERRED_NUMERIC_COLS = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]

# Integer-coded columns that are categories, not quantities.
NUMERIC_AS_CATEGORICAL = ["MSSubClass", "MoSold", "YrSold"]

# Quality / condition scales share one vocabulary in data_description.txt.
# Absence labels introduced by the cleaning rules and "Unknown" need explicit positions:
# absence is ranked below "Po" (no feature < poor feature); "Unknown" is mapped to NaN and
# then imputed in-fold, so the model never sees an arbitrary rank for it.
QUALITY_SCALE = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ABSENCE_LABELS = {
    "NoBasement", "NoGarage", "NoPool", "NoFireplace", "NoMiscFeature", "NoAlleyAccess", "NoFence",
}
UNKNOWN_LABEL = "Unknown"

# ---------------------------------------------------------------------------
# Dictionary-vs-data spelling differences (verified against the CSVs on 2026-11-11)
# Keys are the spellings in data_description.txt, values the spellings in the data.
# ---------------------------------------------------------------------------
DICTIONARY_TO_DATA_COLUMNS = {"Bedroom": "BedroomAbvGr", "Kitchen": "KitchenAbvGr"}
DICTIONARY_TO_DATA_VALUES = {
    "MSZoning": {"C": "C (all)"},
    "Exterior2nd": {"BrkComm": "Brk Cmn", "WdShing": "Wd Shng", "CemntBd": "CmentBd"},
    "Neighborhood": {"Names": "NAmes"},
}

# Candidate outliers investigated in the original notebook (EDA-R01/R02). Any removal is a
# TRAINING-FOLD-ONLY policy evaluated by cross-validation; validation folds are never filtered.
CANDIDATE_OUTLIER_IDS = [524, 1299]
