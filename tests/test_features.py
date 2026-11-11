"""Tests for features.py and preprocess.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from houseprices import cleaning, config, features, preprocess


@pytest.fixture(scope="module")
def frames():
    train, test, _ = cleaning.clean_train_test()
    return train, test


def test_group_names_and_unknown_group(frames):
    train, _ = frames
    assert set(features.FEATURE_SETS) == {
        "base", "areas", "baths", "ages", "quality_interactions", "ordinal_quality",
    }
    with pytest.raises(KeyError):
        features.build_features(train, ["nope"])


def test_base_excludes_id_target_and_constant_flags(frames):
    train, test = frames
    for frame in (train, test):
        base = features.build_features(frame, ["base"])
        for col in [config.ID_COL, config.TARGET, *config.CONSTANT_IN_TRAIN_FLAGS]:
            assert col not in base.columns
        assert len(base) == len(frame)
    assert list(features.build_features(train, ["base"]).columns) == list(
        features.build_features(test, ["base"]).columns
    )


def test_build_features_is_pure(frames):
    train, _ = frames
    before = train.copy()
    features.build_features(train, list(features.FEATURE_SETS))
    pd.testing.assert_frame_equal(train, before)


def test_all_groups_shape(frames):
    train, _ = frames
    full = features.build_features(train, list(features.FEATURE_SETS))
    base = features.build_features(train, ["base"])
    # areas 2 + baths 1 + ages 3 + interactions 2 + ordinal 10 = 18 extra columns
    assert full.shape == (len(train), base.shape[1] + 18)


def test_derived_values(frames):
    _, test = frames
    out = features.build_features(test, ["areas", "baths", "ages", "quality_interactions"])
    row = test.index[test["Id"] == 2550][0]
    assert out.loc[row, "HouseAge"] == 0  # sold before built -> clipped to 0
    row = test.index[test["Id"] == 1877][0]
    assert out.loc[row, "RemodAge"] >= 0
    row = test.index[test["Id"] == 2593][0]
    assert np.isnan(out.loc[row, "GarageAge"])  # implausible GarageYrBlt 2207
    first = test.iloc[0]
    assert out.iloc[0]["TotalSF"] == first["TotalBsmtSF"] + first["1stFlrSF"] + first["2ndFlrSF"]
    assert out.iloc[0]["TotalBath"] == (
        first["FullBath"] + 0.5 * first["HalfBath"] + first["BsmtFullBath"] + 0.5 * first["BsmtHalfBath"]
    )
    assert out.iloc[0]["OverallQual_x_GrLivArea"] == first["OverallQual"] * first["GrLivArea"]


def test_ordinal_quality_mapping():
    s = pd.Series(["Ex", "Gd", "TA", "Fa", "Po", "NoPool", "NoBasement", "Unknown", np.nan])
    result = features.quality_to_number(s).tolist()
    assert result[:7] == [5, 4, 3, 2, 1, 0, 0]
    assert np.isnan(result[7]) and np.isnan(result[8])


def test_preprocessor_no_nan_and_unseen_categories(frames):
    train, test = frames
    X = features.build_features(train, list(features.FEATURE_SETS))
    X_test = features.build_features(test, list(features.FEATURE_SETS))
    pre = preprocess.make_preprocessor(X)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()
    # Test has categories never seen in train (e.g. MSSubClass 150, Unknown labels).
    Xt_test = pre.transform(X_test)
    assert Xt_test.shape == (len(X_test), Xt.shape[1])
    assert not np.isnan(Xt_test).any()


def test_fold_fit_then_transform_unseen_fold(frames):
    train, _ = frames
    X = features.build_features(train, ["base", "areas", "ages"])
    train_idx, val_idx = next(KFold(5, shuffle=True, random_state=config.SEED).split(X))
    pre = preprocess.make_preprocessor(X, scale=False)
    pre.fit(X.iloc[train_idx])
    out = pre.transform(X.iloc[val_idx])
    assert out.shape[0] == len(val_idx)
    assert not np.isnan(out).any()


@pytest.mark.parametrize("lotfrontage", ["neighborhood_median", "global_median"])
@pytest.mark.parametrize("scale", [True, False])
def test_option_combinations(frames, lotfrontage, scale):
    train, _ = frames
    X = features.build_features(train, ["base"])
    out = preprocess.make_preprocessor(X, scale=scale, lotfrontage=lotfrontage).fit_transform(X)
    assert not np.isnan(out).any()


def test_group_median_imputer_uses_fit_data_only():
    fit_df = pd.DataFrame({
        "Neighborhood": ["A", "A", "A", "B", "B"],
        "LotFrontage": [10.0, 20.0, np.nan, 100.0, np.nan],
    })
    imputer = preprocess.GroupMedianImputer().fit(fit_df)
    assert imputer.group_medians_ == {"A": 15.0, "B": 100.0}
    assert imputer.global_median_ == 20.0  # median of 10, 20, 100

    other = pd.DataFrame({
        "Neighborhood": ["A", "B", "C", "A"],
        "LotFrontage": [np.nan, np.nan, np.nan, 7.0],
    })
    filled = imputer.transform(other).ravel()
    # A -> 15, B -> 100, unseen C -> global 20; recorded value 7 kept.
    assert filled.tolist() == [15.0, 100.0, 20.0, 7.0]
    # Transforming did not change the learned statistics.
    assert imputer.group_medians_ == {"A": 15.0, "B": 100.0}
    assert imputer.global_median_ == 20.0
