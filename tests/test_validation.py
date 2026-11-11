"""Independent audit tests for houseprices.validation (Agent D).

Synthetic frames are used wherever possible; one test reads the real train.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, RepeatedKFold

from houseprices import config, validation as v


# ---------------------------------------------------------------------------
# fixtures and dummy estimators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 1460
    df = pd.DataFrame(
        {
            "Id": np.arange(1, n + 1),
            "GrLivArea": rng.uniform(500, 3500, n),
            "SalePrice": np.exp(rng.normal(12.0, 0.4, n)).round(),
        }
    )
    df.loc[df.Id == 524, "GrLivArea"] = 4676.0
    df.loc[df.Id == 1299, "GrLivArea"] = 5642.0
    y = v.to_log_target(df["SalePrice"])
    return df[["Id", "GrLivArea"]], y


class RecordingMean(BaseEstimator, RegressorMixin):
    """Predicts the training mean; records the Ids it was fit on (class-level, clone-safe)."""

    fit_ids: list[np.ndarray] = []
    predict_ids: list[np.ndarray] = []

    def fit(self, X, y):
        type(self).fit_ids.append(X["Id"].to_numpy().copy())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        type(self).predict_ids.append(X["Id"].to_numpy().copy())
        return np.full(len(X), self.mean_)


class Memoriser(BaseEstimator, RegressorMixin):
    """Leaky: memorises y by Id. Perfect on rows seen in fit, mean elsewhere."""

    def fit(self, X, y):
        self.mem_ = dict(zip(X["Id"].to_numpy(), np.asarray(y, dtype=float)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.mem_.get(i, self.mean_) for i in X["Id"].to_numpy()])


class NaNPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


# ---------------------------------------------------------------------------
# (a) metric
# ---------------------------------------------------------------------------


def test_rmse_log_hand_computed():
    # errors 0.1, -0.2, 0.3 -> mean sq = (0.01+0.04+0.09)/3 = 0.046666..., sqrt = 0.216024689...
    got = v.rmse_log([1.0, 2.0, 3.0], [1.1, 1.8, 3.3])
    assert got == pytest.approx(np.sqrt(0.14 / 3), abs=1e-12)
    assert got == pytest.approx(0.2160246899, abs=1e-9)


def test_rmse_log_equals_rmsle_dollars_and_roundtrip():
    price = np.array([100_000.0, 150_000.0, 250_000.0, 400_000.0])
    pred = np.array([110_000.0, 140_000.0, 300_000.0, 380_000.0])
    manual = np.sqrt(np.mean((np.log1p(price) - np.log1p(pred)) ** 2))
    assert v.rmsle_dollars(price, pred) == pytest.approx(manual, abs=1e-12)
    assert v.rmse_log(np.log1p(price), np.log1p(pred)) == pytest.approx(manual, abs=1e-12)
    np.testing.assert_allclose(v.from_log_target(v.to_log_target(pd.Series(price))), price)


def test_rmse_log_shape_mismatch_raises():
    with pytest.raises(ValueError):
        v.rmse_log([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# (b) make_cv
# ---------------------------------------------------------------------------


def _splits(cv, X):
    return [(tr.tolist(), va.tolist()) for tr, va in cv.split(X)]


def test_make_cv_seed_determinism(synth):
    X, _ = synth
    assert _splits(v.make_cv(seed=42), X) == _splits(v.make_cv(seed=42), X)
    assert _splits(v.make_cv(seed=42), X) != _splits(v.make_cv(seed=43), X)
    assert _splits(v.make_cv(), X) == _splits(v.make_cv(seed=config.SEED), X)


def test_make_cv_partitions_all_rows_once(synth):
    X, _ = synth
    vas = np.concatenate([va for _, va in v.make_cv().split(X)])
    assert sorted(vas) == list(range(len(X)))


def test_make_cv_repeated_deterministic(synth):
    X, _ = synth
    a = _splits(v.make_cv(n_repeats=3, seed=42), X)
    assert a == _splits(v.make_cv(n_repeats=3, seed=42), X)
    assert len(a) == 15


# ---------------------------------------------------------------------------
# (c) holdout
# ---------------------------------------------------------------------------


def _decile_share_deviation(df, ids):
    dec = pd.qcut(df[config.TARGET], 10, labels=False, duplicates="drop")
    mask = df[config.ID_COL].isin(ids)
    overall = dec.value_counts(normalize=True).sort_index()
    hold = dec[mask].value_counts(normalize=True).reindex(overall.index).fillna(0)
    return float((overall - hold).abs().max())


def test_holdout_deterministic_size_and_stratified_synthetic(synth):
    X, y = synth
    df = X.assign(SalePrice=np.expm1(y).round())
    a = v.make_holdout_ids(df)
    assert a == v.make_holdout_ids(df)
    assert a != v.make_holdout_ids(df, seed=7)
    assert len(a) == round(0.15 * len(df)) == len(set(a))
    assert _decile_share_deviation(df, a) < 0.02


def test_holdout_real_train_csv():
    df = pd.read_csv(config.TRAIN_CSV)
    ids = v.make_holdout_ids(df)
    assert ids == v.make_holdout_ids(df)
    assert len(ids) == round(0.15 * len(df)) == 219
    assert set(ids) <= set(df[config.ID_COL])
    assert _decile_share_deviation(df, ids) < 0.02
    # row order must not change the answer in terms of which houses are chosen? (it does: documented)
    # => the stored holdout_ids.json, not recomputation, is the source of truth.


def test_split_development_holdout_is_disjoint_partition(synth, tmp_path, monkeypatch):
    X, y = synth
    df = X.assign(SalePrice=np.expm1(y).round())
    monkeypatch.setattr(config, "HOLDOUT_IDS_PATH", tmp_path / "holdout_ids.json")
    dev, hold = v.split_development_holdout(df)
    assert set(dev.Id).isdisjoint(hold.Id)
    assert len(dev) + len(hold) == len(df)
    # second call must read the file, not recompute
    dev2, hold2 = v.split_development_holdout(df.sample(frac=1, random_state=1))
    assert set(hold2.Id) == set(hold.Id)


# ---------------------------------------------------------------------------
# (d) filters only touch training rows
# ---------------------------------------------------------------------------


def test_outlier_filter_trains_without_candidates_but_validation_identical(synth):
    X, y = synth
    cv = v.make_cv()
    RecordingMean.fit_ids, RecordingMean.predict_ids = [], []
    v.cross_validate(RecordingMean(), X, y, cv, v.drop_candidate_outliers)
    fit_ids, pred_ids = RecordingMean.fit_ids, RecordingMean.predict_ids
    assert len(fit_ids) == 5
    for ids in fit_ids:
        assert not set(ids) & {524, 1299}
    # candidates still scored when they fall in a validation fold
    assert {524, 1299} <= set(np.concatenate(pred_ids))

    base = [va.tolist() for _, _, va in v.iter_folds(X, y, cv, v.keep_all)]
    filt = [va.tolist() for _, _, va in v.iter_folds(X, y, cv, v.drop_candidate_outliers)]
    big = [va.tolist() for _, _, va in v.iter_folds(X, y, cv, v.drop_large_living_area(4000))]
    assert base == filt == big
    # training positions: subset of the unfiltered ones, and train/valid disjoint
    for (_, tr0, _), (_, tr1, va) in zip(
        v.iter_folds(X, y, cv, v.keep_all), v.iter_folds(X, y, cv, v.drop_candidate_outliers)
    ):
        assert set(tr1) <= set(tr0)
        assert not set(tr1) & set(va)


def test_filter_receives_only_training_rows(synth):
    X, y = synth
    seen = []

    def spy(Xtr, ytr):
        seen.append(set(Xtr["Id"]))
        return np.ones(len(Xtr), dtype=bool)

    cv = v.make_cv()
    folds = list(v.iter_folds(X, y, cv, spy))
    for ids, (_, tr, va) in zip(seen, folds):
        assert ids == set(X["Id"].iloc[tr])
        assert not ids & set(X["Id"].iloc[va])


def test_filter_wrong_shape_raises(synth):
    X, y = synth
    with pytest.raises(ValueError):
        list(v.iter_folds(X, y, v.make_cv(), lambda a, b: np.ones(3, dtype=bool)))


# ---------------------------------------------------------------------------
# (e) leakage detection
# ---------------------------------------------------------------------------


def test_memoriser_perfect_on_train_but_not_in_cv(synth):
    X, y = synth
    m = Memoriser().fit(X, y)
    assert v.rmse_log(y, m.predict(X)) == pytest.approx(0.0, abs=1e-12)  # leak is real
    res = v.cross_validate(Memoriser(), X, y)
    assert min(res.fold_scores) > 0.1  # predicts the mean on unseen rows
    base = v.cross_validate(RecordingMean(), X, y)
    assert res.mean == pytest.approx(base.mean, rel=1e-9)  # leaky model gains nothing


def test_runner_never_fits_on_validation_rows(synth):
    X, y = synth
    RecordingMean.fit_ids, RecordingMean.predict_ids = [], []
    v.cross_validate(RecordingMean(), X, y)
    for f, p in zip(RecordingMean.fit_ids, RecordingMean.predict_ids):
        assert not set(f) & set(p)
        assert len(f) + len(p) == len(X)


def test_estimator_is_cloned_per_fold(synth):
    X, y = synth
    est = RecordingMean()
    v.cross_validate(est, X, y)
    assert not hasattr(est, "mean_")  # original never fitted


def test_oof_is_complete_and_matches_fold_scores_single_pass(synth):
    X, y = synth
    res = v.cross_validate(RecordingMean(), X, y)
    assert not np.isnan(res.oof_pred).any()
    assert len(res.fold_scores) == 5


# ---------------------------------------------------------------------------
# (f) NaN handling
# ---------------------------------------------------------------------------


def test_nan_predictions_do_not_pass_silently(synth):
    X, y = synth
    try:
        res = v.cross_validate(NaNPredictor(), X, y)
    except (ValueError, FloatingPointError):
        return
    assert np.isnan(res.mean)  # acceptable: NaN is visible


@pytest.mark.xfail(strict=True, reason="AUDIT F2: runner returns NaN instead of raising; a NaN "
                   "cv_mean can be written to the registry and sort silently (na_position='last')")
def test_nan_predictions_should_raise(synth):
    X, y = synth
    with pytest.raises(ValueError):
        v.cross_validate(NaNPredictor(), X, y)


@pytest.mark.xfail(strict=True, reason="AUDIT F3: no guard for an empty training set after the "
                   "filter; a tolerant estimator (mean of nothing) yields NaN instead of an error")
def test_empty_training_after_filter_fails_loudly(synth):
    X, y = synth
    with pytest.raises(ValueError):
        v.cross_validate(RecordingMean(), X, y, v.make_cv(), lambda a, b: np.zeros(len(a), bool))


# ---------------------------------------------------------------------------
# is_single_pass / RepeatedKFold OOF semantics
# ---------------------------------------------------------------------------


def test_repeated_cv_scores_count(synth):
    X, y = synth
    res = v.cross_validate(RecordingMean(), X, y, v.make_cv(n_repeats=3))
    assert len(res.fold_scores) == 15


@pytest.mark.xfail(strict=True, reason="AUDIT F1: is_single_pass is always True because "
                   "RepeatedKFold has no n_splits attribute; OOF is silently overwritten by the "
                   "last repeat although CVResult documents NaN for repeated CV")
def test_repeated_cv_oof_is_nan_as_documented(synth):
    X, y = synth
    res = v.cross_validate(RecordingMean(), X, y, v.make_cv(n_repeats=3))
    assert np.isnan(res.oof_pred).all()


def test_repeatedkfold_has_no_n_splits_attribute():
    # root cause of F1, pinned so that a sklearn change is noticed
    assert not hasattr(RepeatedKFold(n_splits=5, n_repeats=3), "n_splits")
    assert hasattr(KFold(5), "n_splits")
