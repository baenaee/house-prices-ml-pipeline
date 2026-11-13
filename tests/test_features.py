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
        "base", "areas", "baths", "ages", "quality_interactions", "ordinal_quality", "anomaly_flags",
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
    # areas 2 + baths 1 + ages 3 + interactions 2 + ordinal 10 + flags 3 = 21 extra columns
    assert full.shape == (len(train), base.shape[1] + 21)


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


# ---------------------------------------------------------------------------
# A4: anomaly flags group
# ---------------------------------------------------------------------------


def test_anomaly_flags_group_counts_and_ids(frames):
    train, test = frames
    cols = features.ANOMALY_FLAG_COLS
    # The cleaning entry point itself carries the flags on both frames.
    for frame in (train, test):
        assert all(c in frame.columns for c in cols)
    flags_test = features.build_features(test, ["anomaly_flags"])
    assert list(flags_test.columns) == cols
    assert all(str(t) == "int64" for t in flags_test.dtypes)
    assert flags_test.sum().to_dict() == {
        "GarageYrBltImplausible": 2, "SoldBeforeBuilt": 1, "RemodBeforeBuilt": 1,
    }
    by_id = flags_test.set_index(test["Id"])
    assert by_id.loc[2593, "GarageYrBltImplausible"] == 1
    assert by_id.loc[2550, "SoldBeforeBuilt"] == 1
    assert by_id.loc[1877, "RemodBeforeBuilt"] == 1
    # Id 2550 (sold before built) also has an implausible garage year: exactly 3 distinct houses.
    assert int(flags_test.any(axis=1).sum()) == 3
    # The training data has no anomaly at all: the group is constant (all zero) there.
    assert int(features.build_features(train, ["anomaly_flags"]).to_numpy().sum()) == 0


def test_anomaly_flags_never_in_base(frames):
    train, _ = frames
    base = features.build_features(train, ["base"])
    assert not set(features.ANOMALY_FLAG_COLS) & set(base.columns)


# ---------------------------------------------------------------------------
# SkewLog1pTransformer
# ---------------------------------------------------------------------------


def _skew_frame(rng, n=400):
    return pd.DataFrame({
        "skewed": rng.exponential(2.0, n),
        "normal": rng.normal(10, 1, n),
        "negative_skewed": rng.exponential(2.0, n) - 1.0,   # min < 0 -> never selected
        "constant": np.ones(n),
    })


def test_skew_selects_by_skew_and_min():
    rng = np.random.default_rng(0)
    X = _skew_frame(rng)
    t = preprocess.SkewLog1pTransformer(0.75).fit(X)
    assert t.skewed_columns_ == ["skewed"]


def test_skew_columns_depend_on_fold_sample():
    rng = np.random.default_rng(1)
    # Same column, but heavy tail only in the first half: skewed there, symmetric in the second.
    half_a = np.concatenate([rng.normal(5, 0.5, 190), [60.0] * 10])
    half_b = rng.normal(5, 0.5, 200)
    A = pd.DataFrame({"x": half_a, "y": rng.exponential(1, 200)})
    B = pd.DataFrame({"x": half_b, "y": rng.exponential(1, 200)})
    ta = preprocess.SkewLog1pTransformer().fit(A)
    tb = preprocess.SkewLog1pTransformer().fit(B)
    assert "x" in ta.skewed_columns_ and "x" not in tb.skewed_columns_


def test_skew_transform_uses_fitted_columns_on_unseen_rows():
    rng = np.random.default_rng(2)
    X = _skew_frame(rng)
    t = preprocess.SkewLog1pTransformer().fit(X)
    unseen = pd.DataFrame({"skewed": [0.0, 3.0, np.nan], "normal": [1.0, 2.0, 3.0],
                           "negative_skewed": [-5.0, 0.0, 1.0], "constant": [1.0, 1.0, 1.0]})
    out = t.transform(unseen)
    np.testing.assert_allclose(out[:2, 0], np.log1p([0.0, 3.0]))
    assert np.isnan(out[2, 0])
    np.testing.assert_array_equal(out[:, 1:], unseen.to_numpy()[:, 1:])  # untouched columns
    # a unseen sample that is symmetric does not change which columns are used
    sym = pd.DataFrame(rng.normal(0, 1, (50, 4)), columns=X.columns)
    t2 = preprocess.SkewLog1pTransformer().fit(X)
    t2.transform(sym)
    assert t2.skewed_columns_ == ["skewed"]


def test_skew_threshold_edges():
    rng = np.random.default_rng(3)
    X = _skew_frame(rng)
    s = X["skewed"].skew()
    assert preprocess.SkewLog1pTransformer(threshold=s - 1e-9).fit(X).skewed_columns_ == ["skewed"]
    assert preprocess.SkewLog1pTransformer(threshold=s).fit(X).skewed_columns_ == []  # strictly greater
    assert preprocess.SkewLog1pTransformer(threshold=1e9).fit(X).skewed_columns_ == []
    allcols = preprocess.SkewLog1pTransformer(threshold=-1e9).fit(X).skewed_columns_
    assert allcols == ["skewed", "normal", "constant"]  # still excludes min<0 (constant has skew 0)


def test_skew_all_nan_column_is_ignored():
    X = pd.DataFrame({"a": [np.nan] * 5, "b": [0.0, 1.0, 1.0, 1.0, 50.0]})
    t = preprocess.SkewLog1pTransformer().fit(X)
    assert t.skewed_columns_ == ["b"]


def test_make_preprocessor_skew_option(frames):
    train, _ = frames
    X = features.build_features(train, ["base"])
    pre = preprocess.make_preprocessor(X, skew_log1p=True)
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()
    chosen = pre.named_transformers_["numeric"].named_steps["skew"].skewed_columns_
    assert "LotArea" in chosen and "OverallQual" not in chosen
    plain = preprocess.make_preprocessor(X).fit_transform(X)
    assert out.shape == plain.shape


# ---------------------------------------------------------------------------
# KNN LotFrontage option
# ---------------------------------------------------------------------------


def test_knn_lotfrontage_option(frames):
    train, test = frames
    X = features.build_features(train, ["base"])
    X_test = features.build_features(test, ["base"])
    pre = preprocess.make_preprocessor(X, lotfrontage="knn")
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()
    assert not np.isnan(pre.transform(X_test)).any()
    assert pre.fit_transform(X).shape == preprocess.make_preprocessor(X).fit_transform(X).shape
    with pytest.raises(ValueError):
        preprocess.make_preprocessor(X, lotfrontage="mode")


# ---------------------------------------------------------------------------
# YearRangeClipper (opt-in guard for impossible years at prediction time)
# ---------------------------------------------------------------------------

ALL_SET = "base+areas+baths+ages+quality_interactions+ordinal_quality"
E056_KWARGS = {"alpha": 0.0005623413251903491, "scaler": "robust"}


@pytest.fixture(scope="module")
def dev_all():
    from houseprices import data
    return data.load_development_data(ALL_SET, with_holdout_split=True)


def test_year_clipper_fit_uses_fit_rows_only():
    X = pd.DataFrame({"YearBuilt": [1950.0, 1960, 1970, 2000], "GarageYrBlt": [np.nan, 1955, 1965, 2100],
                      "LotArea": [1.0, 2, 3, 4]})
    full = preprocess.YearRangeClipper().fit(X)
    part = preprocess.YearRangeClipper().fit(X.iloc[:3])
    assert full.ranges_ == {"YearBuilt": (1950.0, 2000.0), "GarageYrBlt": (1955.0, 2100.0)}
    assert part.ranges_ == {"YearBuilt": (1950.0, 1970.0), "GarageYrBlt": (1955.0, 1965.0)}
    out = part.transform(X)
    assert out["YearBuilt"].tolist() == [1950, 1960, 1970, 1970]
    low = preprocess.YearRangeClipper().fit(X.iloc[1:])
    assert low.transform(X)["YearBuilt"].tolist()[0] == 1950             # default: lower bound not applied
    assert preprocess.YearRangeClipper(lower=True).fit(X.iloc[1:]).transform(X)["YearBuilt"].tolist()[0] == 1960
    assert np.isnan(out.loc[0, "GarageYrBlt"]) and out.loc[3, "GarageYrBlt"] == 1965
    assert out["LotArea"].tolist() == X["LotArea"].tolist()          # non-year columns untouched
    assert X.loc[3, "YearBuilt"] == 2000                              # input not modified


def test_year_clipper_ignores_missing_year_columns():
    X = pd.DataFrame({"LotArea": [1.0, 2.0]})
    assert preprocess.YearRangeClipper().fit(X).ranges_ == {}


def test_year_clipper_is_noop_on_development_rows(dev_all):
    from houseprices import models
    X = dev_all.X
    numeric, _ = preprocess.split_columns(X.drop(columns=["Id"]))
    clip = preprocess.YearRangeClipper().fit(X[numeric])
    assert set(clip.ranges_) == {"YearBuilt", "YearRemodAdd", "GarageYrBlt"}   # YrSold is categorical here
    pd.testing.assert_frame_equal(clip.transform(X[numeric]), X[numeric].astype("float64"),
                                  check_dtype=False)
    base = models.build_lasso(**E056_KWARGS, preprocessor_kwargs={"skew_log1p": True})
    clipped = models.build_lasso(**E056_KWARGS, preprocessor_kwargs={"skew_log1p": True, "clip_years": True})
    base.fit(X, dev_all.y_log)
    clipped.fit(X, dev_all.y_log)
    np.testing.assert_array_equal(base.predict(X), clipped.predict(X))


def test_year_clipper_on_test_changes_only_garage_2207(dev_all):
    """Upper bound only (D20): Id 2593 2207 -> training max; the 1895/1896 garages are left alone."""
    X, T = dev_all.X, dev_all.X_test
    numeric, _ = preprocess.split_columns(X.drop(columns=["Id"]))
    clip = preprocess.YearRangeClipper().fit(X[numeric])
    before, after = T[numeric].astype("float64"), clip.transform(T[numeric])
    changed = before.ne(after) & ~(before.isna() & after.isna())
    rows, cols = np.nonzero(changed.to_numpy())
    found = {(int(T["Id"].iloc[r]), numeric[c], before.iloc[r, c], after.iloc[r, c]) for r, c in zip(rows, cols)}
    assert found == {(2593, "GarageYrBlt", 2207.0, 2010.0)}
    assert after["GarageYrBlt"].max() == X["GarageYrBlt"].max()
    # the optional two-sided version also lifts the two garages older than the training minimum
    both = preprocess.YearRangeClipper(lower=True).fit(X[numeric]).transform(T[numeric])
    assert int((before.ne(both) & ~(before.isna() & both.isna())).to_numpy().sum()) == 3


def test_make_preprocessor_clip_years_option(dev_all):
    X = dev_all.X.drop(columns=["Id"])
    pre = preprocess.make_preprocessor(X, clip_years=True, skew_log1p=True)
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()
    assert [n for n, _ in pre.named_transformers_["numeric"].steps][:2] == ["clip_years", "skew"]
    assert [n for n, _ in preprocess.make_preprocessor(X).fit(X).named_transformers_["numeric"].steps] == ["impute", "scale"]
