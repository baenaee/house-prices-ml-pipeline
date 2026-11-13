"""Smoke tests for every builder in ``houseprices.models``.

They fit on a small sample of the real cleaned development columns (no holdout split, no
scores recorded) and check shape, finiteness and that ``sklearn.clone`` round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from houseprices import data, models

SMALL = dict(n_estimators=15, max_depth=3)

BUILD_CASES = {
    "mean": lambda: models.build_mean_baseline(),
    "ols_numeric": lambda: models.build_ols_numeric(),
    "ridge": lambda: models.build_ridge(alpha=10.0),
    "ridge_standard": lambda: models.build_ridge(alpha=10.0, scaler="standard"),
    "lasso": lambda: models.build_lasso(alpha=0.01),
    "elasticnet": lambda: models.build_elasticnet(alpha=0.01, l1_ratio=0.5),
    "xgboost": lambda: models.build_xgboost(**SMALL),
    "lightgbm": lambda: models.build_lightgbm(n_estimators=15, num_leaves=7, min_child_samples=5),
    "catboost_onehot": lambda: models.build_catboost(iterations=15, depth=3),
    "catboost_native": lambda: models.build_catboost(iterations=15, depth=3, native_categoricals=True),
}


@pytest.fixture(scope="module")
def sample():
    dev = data.load_development_data("base", with_holdout_split=False)
    return dev.X.head(150), dev.y_log.head(150), dev.X.iloc[150:200]


def test_all_builders_build_without_data():
    for name, make in BUILD_CASES.items():
        assert make() is not None, name


@pytest.mark.parametrize("name", list(BUILD_CASES))
def test_fit_predict_finite_and_shaped(name, sample):
    X, y, X_new = sample
    est = BUILD_CASES[name]().fit(X, y)
    for frame in (X, X_new):                      # training rows and unseen rows
        pred = np.asarray(est.predict(frame))
        assert pred.shape == (len(frame),)
        assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("name", list(BUILD_CASES))
def test_clone_round_trips_and_is_unfitted(name, sample):
    X, y, _ = sample
    est = BUILD_CASES[name]()
    twin = clone(est)
    assert repr(twin.get_params(deep=False).keys()) == repr(est.get_params(deep=False).keys())
    # deep params of every step must match (covers LazyPreprocessor and CatBoostNative)
    a, b = est.get_params(deep=True), twin.get_params(deep=True)
    def scalar(v):
        return isinstance(v, (int, str, bool, type(None))) or (isinstance(v, float) and v == v)

    simple = {k for k, v in a.items() if scalar(v)}
    assert {k: a[k] for k in simple} == {k: b[k] for k in simple}
    # estimator-valued params (steps) must have equal reprs, which covers dict/list contents
    assert repr(a["steps"]) == repr(b["steps"])
    twin.fit(X, y)                                # the clone fits; the original stays unfitted
    assert not hasattr(est.steps[-1][1], "coef_") and not hasattr(est.steps[-1][1], "model_")


def test_clone_of_fitted_lazy_pipeline_is_independent(sample):
    X, y, _ = sample
    est = models.build_ridge().fit(X, y)
    fresh = clone(est)
    assert not hasattr(fresh.named_steps["prep"], "column_transformer_")


def test_native_catboost_receives_text_columns(sample):
    X, y, _ = sample
    est = models.build_catboost(iterations=10, depth=3, native_categoricals=True).fit(X, y)
    cat_cols = est.named_steps["model"].cat_cols_
    assert "Neighborhood" in cat_cols and "GrLivArea" not in cat_cols


def test_id_column_is_not_a_feature(sample):
    X, y, _ = sample
    est = models.build_ridge().fit(X, y)
    assert "Id" not in list(est.named_steps["prep"].get_feature_names_out())


def test_scaler_option_changes_the_scaler(sample):
    X, y, _ = sample
    robust = models.build_ridge(scaler="robust").fit(X, y)
    standard = models.build_ridge(scaler="standard").fit(X, y)
    ct_r = robust.named_steps["prep"].column_transformer_.named_transformers_["numeric"]
    ct_s = standard.named_steps["prep"].column_transformer_.named_transformers_["numeric"]
    assert type(ct_r.named_steps["scale"]).__name__ == "RobustScaler"
    assert type(ct_s.named_steps["scale"]).__name__ == "StandardScaler"


def test_invalid_scaler_rejected(sample):
    X, y, _ = sample
    with pytest.raises(ValueError):
        models.build_ridge(scaler="minmax").fit(X, y)


def test_optuna_spaces_return_complete_params():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name, space in models.OPTUNA_SPACES.items():
        study = optuna.create_study()
        captured = {}

        def objective(t, space=space, captured=captured):
            captured["p"] = space(t)
            return 0.0

        study.optimize(objective, n_trials=1)
        params = captured["p"]
        assert set(study.best_trial.params) <= set(params)
        if name == "lightgbm":
            assert params["subsample_freq"] == 1 and "subsample_freq" not in study.best_trial.params
