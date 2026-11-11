"""Tests for houseprices.ensemble using synthetic data (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from houseprices import ensemble, validation


@pytest.fixture(scope="module")
def xy():
    X, y = make_regression(n_samples=300, n_features=8, n_informative=5, noise=10.0, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


def bases():
    return [
        ("ridge", Ridge(alpha=1.0)),
        ("tree", DecisionTreeRegressor(max_depth=4, random_state=0)),
        ("knn", KNeighborsRegressor(n_neighbors=7)),
    ]


class Memoriser(BaseEstimator, RegressorMixin):
    """Returns the exact target for rows it was trained on and the mean for unseen rows."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.table_ = {tuple(r): v for r, v in zip(X, np.asarray(y, dtype=float))}
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.table_.get(tuple(r), self.mean_) for r in np.asarray(X, dtype=float)])


# (a) the meta-learner never scores rows it was fit on --------------------------------


def test_nested_stack_is_not_fooled_by_a_memorising_base(xy):
    X, y = xy
    mem = [("mem", Memoriser())]
    meta = Ridge(alpha=1e-6)
    stack = ensemble.NestedStackingRegressor(mem, meta)
    nested = validation.cross_validate(stack, X, y).mean
    leaky = ensemble.in_sample_stack_score(mem, meta, X, y)
    assert leaky < 1e-6                     # leak: memoriser looks perfect on its own rows
    assert nested > 0.8 * float(np.std(y))  # nested: it is no better than predicting the mean


def test_inner_oof_rows_are_never_seen_by_the_model_that_predicts_them(xy):
    X, y = xy
    oof = ensemble.inner_oof_matrix([("mem", Memoriser())], X, y, n_splits=5, seed=42)
    assert oof.shape == (len(y), 1) and not np.isnan(oof).any()
    # a memoriser would return y exactly for seen rows; OOF values are all fold means instead
    assert np.abs(oof[:, 0] - y.to_numpy()).mean() > 0.5 * float(np.std(y))


def test_optimistic_oof_diagnostic_is_not_larger_than_nested(xy):
    X, y = xy
    nested = validation.cross_validate(ensemble.NestedStackingRegressor(bases(), Ridge()), X, y).mean
    optimistic = ensemble.optimistic_oof_stack_score(bases(), Ridge(), X, y)
    assert optimistic <= nested + 1e-9


# (b) / (c) blend weights --------------------------------------------------------------


def test_blend_weights_nonnegative_and_sum_to_one(xy):
    X, y = xy
    oof = ensemble.inner_oof_matrix(bases(), X, y, 5, 42)
    w = ensemble.fit_blend_weights(oof, y.to_numpy())
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    w_nn = ensemble.fit_blend_weights(oof, y.to_numpy(), non_negative=True, sum_to_one=False)
    assert (w_nn >= 0).all()
    w_free = ensemble.fit_blend_weights(oof, y.to_numpy(), non_negative=False, sum_to_one=True)
    assert w_free.sum() == pytest.approx(1.0)


def test_blend_weights_recover_known_mixture():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    A = np.column_stack([y + rng.normal(scale=0.3, size=500), rng.normal(size=500)])
    A[:, 1] = y + rng.normal(scale=0.3, size=500)
    target = 0.7 * A[:, 0] + 0.3 * A[:, 1]
    w = ensemble.fit_blend_weights(A, target)
    assert w == pytest.approx([0.7, 0.3], abs=1e-4)


def test_learned_blend_matches_manual_computation(xy):
    X, y = xy
    blend = ensemble.BlendingRegressor(bases(), weights=None, inner_cv_splits=5, seed=42).fit(X, y)
    oof = ensemble.inner_oof_matrix(bases(), X, y, 5, 42)
    manual_w = ensemble.fit_blend_weights(oof, y.to_numpy())
    np.testing.assert_allclose(blend.weights_, manual_w)
    manual_pred = np.column_stack(
        [clone(est).fit(X, y).predict(X) for _, est in bases()]
    ) @ manual_w
    np.testing.assert_allclose(blend.predict(X), manual_pred)


def test_fixed_weight_blend_is_weighted_average(xy):
    X, y = xy
    blend = ensemble.BlendingRegressor(bases()[:2], weights=[0.25, 0.75]).fit(X, y)
    expected = 0.25 * clone(bases()[0][1]).fit(X, y).predict(X) + 0.75 * clone(bases()[1][1]).fit(X, y).predict(X)
    np.testing.assert_allclose(blend.predict(X), expected)


# (d) sklearn compatibility ------------------------------------------------------------


def test_clone_and_params_roundtrip(xy):
    X, y = xy
    stack = ensemble.NestedStackingRegressor(bases(), Ridge(alpha=2.0), inner_cv_splits=4, seed=7, passthrough=True)
    blend = ensemble.BlendingRegressor(bases(), weights=[0.2, 0.3, 0.5])
    for est in (stack, blend):
        c = clone(est)
        assert c is not est
        assert set(c.get_params(deep=False)) == set(est.get_params(deep=False))
    c = clone(stack)
    assert (c.inner_cv_splits, c.seed, c.passthrough) == (4, 7, True)
    assert c.meta_estimator.alpha == 2.0 and c.meta_estimator is not stack.meta_estimator
    c.set_params(inner_cv_splits=3, passthrough=False)
    assert c.inner_cv_splits == 3 and stack.inner_cv_splits == 4
    stack.fit(X, y)                       # fitting the original must not alter its parameters
    assert [n for n, _ in stack.base_estimators] == ["ridge", "tree", "knn"]
    assert not hasattr(stack.base_estimators[0][1], "coef_")


def test_passthrough_adds_raw_features(xy):
    X, y = xy
    stack = ensemble.NestedStackingRegressor(bases(), Ridge(), passthrough=True).fit(X, y)
    assert stack.meta_.coef_.shape[0] == 3 + X.shape[1]
    assert stack.predict(X).shape == (len(X),)


# (e) determinism ------------------------------------------------------------------------


def test_same_seed_same_result_and_seed_matters(xy):
    X, y = xy
    run = lambda seed: validation.cross_validate(  # noqa: E731
        ensemble.NestedStackingRegressor(bases(), Ridge(), seed=seed), X, y
    ).fold_scores
    assert run(42) == run(42)
    assert run(42) != run(1)


def test_filter_applies_to_training_side_only(xy):
    X, y = xy
    seen = {}

    class Spy(Ridge):
        def fit(self, X, y, **kw):
            seen.setdefault("n", []).append(len(y))
            return super().fit(X, y, **kw)

    drop_first_20 = lambda Xt, yt: np.arange(len(Xt)) >= 20  # noqa: E731
    res = validation.cross_validate(
        ensemble.NestedStackingRegressor([("spy", Spy())], Ridge()), X, y, train_row_filter=drop_first_20
    )
    assert res.n_train_rows_per_fold == [220] * 5      # 240 outer-train rows minus 20 filtered
    assert max(seen["n"]) <= 220                        # no base ever sees more than filtered train


def test_diversity_report_and_correlation(xy):
    X, y = xy
    oof = pd.DataFrame(ensemble.inner_oof_matrix(bases(), X, y, 5, 42), columns=["ridge", "tree", "knn"])
    corr = ensemble.prediction_correlation(oof)
    assert corr.shape == (3, 3) and np.allclose(np.diag(corr), 1.0)
    rep = ensemble.diversity_report(oof, y)
    assert list(rep.index) == ["ridge", "tree", "knn"]
    assert (rep["rmse"] > 0).all()
