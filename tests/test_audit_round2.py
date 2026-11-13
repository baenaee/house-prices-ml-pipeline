"""Agent D (validation-auditor) round 2: executable probes for cleaning, features, preprocess,
models, data entry point and the experiment runner.

Probes assert verified behaviour. A defect is pinned with ``xfail(strict=True, reason="F<id>")``.
The real holdout file is never created: every real-data test redirects HOLDOUT_IDS_PATH (and the
registry paths) to a temporary directory.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate as sk_cross_validate

from houseprices import cleaning, config, data, features, models, preprocess, validation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_holdout(tmp_path_factory):
    """Redirect the holdout file to a temp location for the whole module."""
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "HOLDOUT_IDS_PATH", tmp_path_factory.mktemp("holdout") / "holdout_ids.json")
    yield config.HOLDOUT_IDS_PATH
    mp.undo()


@pytest.fixture(scope="module")
def loaded():
    train, raw_train = cleaning.load_raw(config.TRAIN_CSV)
    test, raw_test = cleaning.load_raw(config.TEST_CSV)
    return train, raw_train, test, raw_test


@pytest.fixture(scope="module")
def cleaned_pair():
    train_c, test_c, counts = cleaning.clean_train_test()
    return train_c, test_c, counts


@pytest.fixture(scope="module")
def dev(tmp_holdout):
    return data.load_development_data("base")


def _source(module) -> str:
    return inspect.getsource(module)


def _code_names(module) -> tuple[set[str], set[str]]:
    """(attribute names, string constants outside docstrings) used by a module."""
    tree = ast.parse(_source(module))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))
    attrs, consts = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Name):
            attrs.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            consts.add(node.value)
    return attrs, consts


# ---------------------------------------------------------------------------
# 1. cleaning.py
# ---------------------------------------------------------------------------


def test_cleaning_never_reads_target_or_fits_statistics():
    attrs, consts = _code_names(cleaning)
    assert "SalePrice" not in consts and "TARGET" not in attrs
    fitted = {"median", "mean", "mode", "std", "var", "quantile", "groupby", "transform", "interpolate",
              "ffill", "bfill", "corr", "rank", "describe", "concat", "merge", "append", "fit", "fit_transform"}
    assert not (fitted & attrs), fitted & attrs
    # every fillna argument is a literal constant
    tree = ast.parse(_source(cleaning))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fillna":
            assert isinstance(node.args[0], ast.Constant), ast.dump(node)


def test_cleaning_has_no_dtype_dependent_logic():
    src = _source(cleaning)
    assert "select_dtypes" not in src and "== object" not in src and "dtype" not in src


def test_clean_is_row_wise_so_train_and_test_cannot_influence_each_other(loaded):
    """clean(subset) == clean(full)[subset]: no rule uses information from other rows."""
    train, raw_train, test, raw_test = loaded
    for df, raw in ((train, raw_train), (test, raw_test)):
        full, _ = cleaning.clean(df, raw)
        rng = np.random.default_rng(0)
        pos = np.sort(rng.choice(len(df), 300, replace=False))
        # include the interesting test row when present
        sub, _ = cleaning.clean(df.iloc[pos], raw)
        pd.testing.assert_frame_equal(sub, full.iloc[pos], check_dtype=True)


def test_train_and_test_use_same_function_and_same_columns(cleaned_pair):
    train_c, test_c, _ = cleaned_pair
    assert set(train_c.columns) - set(test_c.columns) == {config.TARGET}
    assert list(train_c.columns.drop(config.TARGET)) == list(test_c.columns)


def test_b04_is_a_condition_selecting_only_test_2121(loaded):
    train, _, test, _ = loaded
    assert not cleaning.basement_fully_unrecorded(train).any()
    assert test.loc[cleaning.basement_fully_unrecorded(test), "Id"].tolist() == [2121]
    for fn in (cleaning.rule_b04, cleaning.basement_fully_unrecorded):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 2121]
        assert '"Id"' not in "".join(inspect.getsource(fn).split('"""')[::2])


def test_c01_is_labelled_reconstruction():
    assert "RECONSTRUCT" in cleaning.rule_c01.__doc__.upper()
    assert "NOT recovered" in cleaning.rule_c01.__doc__
    assert "reconstructed" in _source(config).lower()


def test_clean_leaves_inputs_untouched_and_preserves_order(loaded):
    train, raw_train, *_ = loaded
    before, before_raw = train.copy(deep=True), raw_train.copy(deep=True)
    out, _ = cleaning.clean(train, raw_train)
    pd.testing.assert_frame_equal(train, before)
    pd.testing.assert_frame_equal(raw_train, before_raw)
    assert out["Id"].tolist() == train["Id"].tolist()
    pd.testing.assert_series_equal(out[config.TARGET], train[config.TARGET])


def test_clean_train_test_only_adds_anomaly_columns_and_never_edits_values(loaded, cleaned_pair):
    train, raw_train, test, raw_test = loaded
    train_c, test_c, _ = cleaned_pair
    for df, raw, out in ((train, raw_train, train_c), (test, raw_test, test_c)):
        plain, _ = cleaning.clean(df, raw)
        extra = [c for c in out.columns if c not in plain.columns]
        assert sorted(extra) == sorted(features.ANOMALY_FLAG_COLS)
        pd.testing.assert_frame_equal(out[plain.columns], plain)


def test_anomaly_flag_targets(cleaned_pair):
    _, test_c, _ = cleaned_pair
    t = test_c.set_index("Id")
    assert t.loc[2593, "GarageYrBltImplausible"] and t.loc[2550, "SoldBeforeBuilt"] and t.loc[1877, "RemodBeforeBuilt"]
    assert t.loc[2593, "GarageYrBlt"] == 2207  # raw value preserved


def test_no_object_nan_after_cleaning(cleaned_pair):
    for frame in cleaned_pair[:2]:
        text = frame.select_dtypes(include=["object", "string"])
        assert text.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 2. features.py and the Id question (F9)
# ---------------------------------------------------------------------------

ALL_GROUPS = list(features.FEATURE_SETS)


def test_every_feature_group_is_row_wise(cleaned_pair):
    train_c, _, _ = cleaned_pair
    full = features.build_features(train_c, ALL_GROUPS[:1] + [g for g in ALL_GROUPS[1:]])
    rng = np.random.default_rng(1)
    pos = np.sort(rng.choice(len(train_c), 200, replace=False))
    sub = features.build_features(train_c.iloc[pos], ALL_GROUPS)
    pd.testing.assert_frame_equal(sub, full.iloc[pos])


def test_feature_groups_never_contain_target(cleaned_pair):
    train_c, test_c, _ = cleaned_pair
    X = features.build_features(train_c, ALL_GROUPS)
    assert config.TARGET not in X.columns and not any("SalePrice" in c for c in X.columns)
    attrs, consts = _code_names(features)
    assert "SalePrice" not in consts
    assert not ({"median", "mean", "groupby", "fit", "fit_transform", "quantile", "mode"} & attrs)


def test_quality_to_number_unknown_is_nan():
    s = pd.Series(["Po", "Fa", "TA", "Gd", "Ex", "NoBasement", "NoGarage", "Unknown", np.nan, "weird"])
    out = features.quality_to_number(s)
    assert out.tolist()[:7] == [1, 2, 3, 4, 5, 0, 0]
    assert out.iloc[7:].isna().all()
    assert out.dtype == "float64"


def test_build_features_guards(cleaned_pair):
    train_c, _, _ = cleaned_pair
    with pytest.raises(ValueError):
        features.build_features(train_c, ["base", "base"])
    with pytest.raises(KeyError):
        features.build_features(train_c, ["nope"])


def test_base_group_excludes_id_by_design_and_data_reattaches_it(cleaned_pair, dev):
    """F9 (fixed by director): group_base drops Id; load_development_data re-attaches it first."""
    train_c, _, _ = cleaned_pair
    assert "Id" not in features.build_features(train_c, ["base"]).columns
    for frame in (dev.X, dev.X_holdout, dev.X_test):
        assert frame.columns[0] == "Id"
    keep = validation.drop_candidate_outliers(dev.X, dev.y_log)
    assert (~keep).sum() == 2


def test_x_has_no_target_and_no_anomaly_flags_in_base(dev):
    for frame in (dev.X, dev.X_holdout, dev.X_test):
        assert config.TARGET not in frame.columns
        assert not set(features.ANOMALY_FLAG_COLS) & set(frame.columns)
    assert list(dev.X.columns) == list(dev.X_test.columns) == list(dev.X_holdout.columns)


def test_anomaly_flags_group_is_extra_and_numeric(tmp_holdout):
    d = data.load_development_data("base+anomaly_flags")
    assert set(features.ANOMALY_FLAG_COLS) <= set(d.X.columns)
    assert all(pd.api.types.is_integer_dtype(d.X[c]) for c in features.ANOMALY_FLAG_COLS)


# ---------------------------------------------------------------------------
# 3. preprocess.py
# ---------------------------------------------------------------------------


def _frame(lot, hood):
    return pd.DataFrame({"LotFrontage": lot, "Neighborhood": hood})


def test_group_median_imputer_learns_only_in_fit_from_given_rows():
    a = _frame([10.0, 20.0, 30.0, np.nan], ["N", "N", "N", "N"])
    b = _frame([100.0, 200.0, 300.0, np.nan], ["N", "N", "N", "S"])
    ia, ib = preprocess.GroupMedianImputer().fit(a), preprocess.GroupMedianImputer().fit(b)
    assert ia.group_medians_["N"] == 20.0 and ib.group_medians_["N"] == 200.0
    assert ia.global_median_ == 20.0 and ib.global_median_ == 200.0
    new = _frame([np.nan, np.nan, 5.0], ["N", "Z", "N"])  # Z unseen -> global median of training rows
    out_a = ia.transform(new).ravel()
    assert out_a.tolist() == [20.0, 20.0, 5.0]
    assert ib.transform(new).ravel().tolist() == [200.0, 200.0, 5.0]
    # transform must not change the fitted state or learn from the new rows
    assert ia.group_medians_["N"] == 20.0 and ia.global_median_ == 20.0


def test_group_median_imputer_all_nan_group_falls_back_to_global():
    tr = _frame([10.0, 30.0, np.nan], ["A", "A", "B"])
    imp = preprocess.GroupMedianImputer().fit(tr)
    assert "B" not in imp.group_medians_
    assert imp.transform(_frame([np.nan], ["B"])).ravel().tolist() == [20.0]


def test_make_preprocessor_reads_only_names_and_dtypes(dev):
    X = dev.X.drop(columns="Id")
    shuffled = X.sample(frac=1.0, random_state=0).reset_index(drop=True)
    empty = X.iloc[0:0]
    p_full = preprocess.make_preprocessor(X)
    p_shuf = preprocess.make_preprocessor(shuffled)
    p_empty = preprocess.make_preprocessor(empty)
    assert repr(p_full.get_params()["transformers"]) == repr(p_shuf.get_params()["transformers"])
    assert repr(p_full.get_params()["transformers"]) == repr(p_empty.get_params()["transformers"])


def test_as_text_keeps_nan_and_stringifies_codes():
    X = pd.DataFrame({"a": [60, 20, 60], "b": ["x", None, "y"], "c": [1.0, np.nan, 3.0]})
    out = preprocess._as_text(X)
    assert out["a"].tolist() == ["60", "20", "60"]
    assert pd.isna(out.loc[1, "b"]) and pd.isna(out.loc[1, "c"])
    assert out.loc[0, "c"] == "1.0"


def test_numeric_as_categorical_routing_and_split(dev):
    X = dev.X.drop(columns="Id")
    num, cat = preprocess.split_columns(X)
    assert set(config.NUMERIC_AS_CATEGORICAL) <= set(cat)
    assert not set(config.NUMERIC_AS_CATEGORICAL) & set(num)
    assert set(num) | set(cat) == set(X.columns) and not set(num) & set(cat)
    assert all(X[c].dtype != object for c in num)


def test_onehot_ignores_unseen_categories_and_scaler_only_when_requested(dev):
    X = dev.X.drop(columns="Id")
    y = dev.y_log
    pre = preprocess.make_preprocessor(X, scale=True).fit(X.iloc[:300], y.iloc[:300])
    unseen = X.iloc[300:310].copy()
    unseen["Neighborhood"] = "Atlantis"
    unseen["MSSubClass"] = 9999
    out = pre.transform(unseen)
    assert np.isfinite(out).all()
    for scale in (True, False):
        pre = preprocess.make_preprocessor(X, scale=scale)
        has_scaler = "RobustScaler" in repr(pre.get_params()["transformers"])
        assert has_scaler is scale


def test_preprocessor_transform_is_stateless_for_new_rows(dev):
    X = dev.X.drop(columns="Id")
    pre = preprocess.make_preprocessor(X).fit(X.iloc[:500])
    a = pre.transform(X.iloc[500:700])
    pre.transform(X.iloc[700:900])
    b = pre.transform(X.iloc[500:700])
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 4. models.py
# ---------------------------------------------------------------------------


def test_builders_need_no_data_and_fix_seeds():
    for name in ("mean_baseline", "ols_numeric", "ridge", "lasso", "elasticnet", "xgboost", "lightgbm", "catboost"):
        est = getattr(models, f"build_{name}")()
        assert hasattr(est, "fit")
    assert getattr(models.build_ridge().named_steps["model"], "random_state") == 42
    assert models.build_xgboost().named_steps["model"].get_params()["random_state"] == 42
    assert models.build_xgboost().named_steps["model"].get_params()["n_jobs"] == models.N_THREADS
    assert models.build_lightgbm().named_steps["model"].get_params()["random_state"] == 42
    assert models.build_catboost().named_steps["model"].get_params()["random_seed"] == 42
    assert models.build_catboost(native_categoricals=True).named_steps["model"].params in (None, {})


def test_lazy_preprocessor_refits_per_fold_with_training_medians(dev):
    X, y = dev.X, dev.y_log
    est = models.build_ridge(alpha=10.0)
    cv = KFold(5, shuffle=True, random_state=42)
    res = sk_cross_validate(est, X, y, cv=cv, return_estimator=True, scoring="neg_root_mean_squared_error")
    meds, lot = [], []
    for (tr, _), fitted in zip(cv.split(X), res["estimator"]):
        ct = fitted.named_steps["prep"].column_transformer_
        num_cols = ct.transformers_[1][2]
        stats = ct.named_transformers_["numeric"].named_steps["impute"].statistics_
        expected = X.iloc[tr][num_cols].median().to_numpy()
        np.testing.assert_allclose(stats, expected)
        meds.append(stats)
        grp = ct.named_transformers_["lotfrontage"].named_steps["group_median"]
        lf = X.iloc[tr]
        assert grp.global_median_ == pytest.approx(lf["LotFrontage"].median())
        assert grp.group_medians_["NAmes"] == pytest.approx(lf.loc[lf.Neighborhood == "NAmes", "LotFrontage"].median())
        lot.append(grp.group_medians_["NAmes"])
        assert "Id" not in ct.get_feature_names_out()
    assert not np.allclose(meds[0], meds[1])      # medians really differ between folds
    assert len(set(lot)) > 1
    assert "prep" in est.named_steps and not hasattr(est.named_steps["prep"], "column_transformer_")  # original unfitted


def test_numeric_median_imputer_fits_training_median(dev):
    X = dev.X
    tr, va = X.iloc[:800], X.iloc[800:]
    imp = models.NumericMedianImputer().fit(tr)
    num, _ = preprocess.split_columns(tr.drop(columns="Id"))
    assert imp.columns_ == num and "Id" not in imp.columns_
    np.testing.assert_allclose(imp.imputer_.statistics_, tr[num].median().to_numpy())
    imp.transform(va)
    np.testing.assert_allclose(imp.imputer_.statistics_, tr[num].median().to_numpy())


def test_standard_scaler_swap_actually_swaps(dev):
    X = dev.X.iloc[:400]
    lp = models.LazyPreprocessor(scaler="standard").fit(X, dev.y_log.iloc[:400])
    assert "StandardScaler" in repr(lp.column_transformer_.transformers_)
    assert "RobustScaler" not in repr(lp.column_transformer_.transformers_)


def test_catboost_native_fill_text_learns_nothing(dev):
    pytest.importorskip("catboost")
    X, y = dev.X.iloc[:300], dev.y_log.iloc[:300]
    est = models.build_catboost(native_categoricals=True, iterations=20)
    est.fit(X, y)
    cb = est.named_steps["model"]
    before = (list(cb.cat_cols_), repr(cb.__dict__.keys()))
    Xt = est.named_steps["prep"].transform(dev.X.iloc[300:320])
    Xt.iloc[0, Xt.columns.get_loc("Neighborhood")] = np.nan
    filled = cb._fill_text(Xt)
    assert filled.loc[filled.index[0], "Neighborhood"] == "Unknown"
    assert (list(cb.cat_cols_), repr(cb.__dict__.keys())) == before
    assert np.isfinite(est.predict(dev.X.iloc[300:320])).all()


def test_mean_baseline_and_ols_predict_finite(dev):
    X, y = dev.X, dev.y_log
    for est in (models.build_mean_baseline(), models.build_ols_numeric()):
        pred = clone(est).fit(X.iloc[:900], y.iloc[:900]).predict(X.iloc[900:])
        assert np.isfinite(pred).all()


# ---------------------------------------------------------------------------
# 5. data.py
# ---------------------------------------------------------------------------


def test_holdout_split_then_features_equals_features_then_split(tmp_holdout, dev):
    train_c, _, _ = cleaning.clean_train_test()
    whole = features.build_features(train_c, ["base"])
    whole.insert(0, "Id", train_c["Id"].to_numpy())
    hold_ids = set(validation.load_or_create_holdout_ids(train_c))
    mask = train_c["Id"].isin(hold_ids).to_numpy()
    pd.testing.assert_frame_equal(
        whole.loc[~mask].reset_index(drop=True), dev.X.reset_index(drop=True), check_dtype=True
    )
    assert not set(dev.X["Id"]) & set(dev.X_holdout["Id"])
    assert len(dev.X) + len(dev.X_holdout) == 1460
    assert (dev.X.index == dev.y_log.index).all()


def test_y_log_aligned_with_ids(dev):
    train = pd.read_csv(config.TRAIN_CSV).set_index("Id")
    expected = np.log1p(train.loc[dev.X["Id"], "SalePrice"].to_numpy())
    np.testing.assert_allclose(dev.y_log.to_numpy(), expected)


def test_run_cv_never_touches_holdout_frames():
    common = _load_common()
    assert "holdout" not in inspect.getsource(common.run_cv).lower()
    assert "holdout" not in inspect.getsource(common.run_and_register).lower()
    # the only mention is the dry-run pass-through of empty holdout frames in get_data
    assert common._DRY_RUN_ROWS is None


def test_row_filter_names_resolve_consistently(dev):
    f = data.get_train_row_filter("drop_GrLivArea_gt_4000")
    assert f.__name__ == "drop_GrLivArea_gt_4000"
    keep = f(dev.X, dev.y_log)
    assert (dev.X.loc[~keep, "GrLivArea"] > 4000).all() and (dev.X.loc[keep, "GrLivArea"] <= 4000).all()
    assert data.get_train_row_filter("keep_all") is validation.keep_all
    assert data.get_train_row_filter("drop_candidate_outliers") is validation.drop_candidate_outliers
    with pytest.raises(ValueError):
        data.get_train_row_filter("bogus")


# ---------------------------------------------------------------------------
# 6. runner, registry record and evaluate (R6)
# ---------------------------------------------------------------------------


def _load_common():
    path = config.PROJECT_ROOT / "experiments" / "models" / "_common.py"
    spec = importlib.util.spec_from_file_location("audit_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_register_then_evaluate_reproduces_and_duplicate_id_refused(tmp_path, tmp_holdout, monkeypatch):
    from houseprices import evaluate

    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    common = _load_common()
    kwargs = dict(
        title="t", hypothesis="h", rationale="r", implementation="i",
        builder="houseprices.models:build_ridge", builder_kwargs={"alpha": 10.0},
        interpret=lambda r: ("ok", "retain"), feature_set="base", train_row_filter="drop_candidate_outliers",
    )
    exp = common.run_and_register("E901", **kwargs)
    with pytest.raises(ValueError):
        common.run_and_register("E901", **kwargs)
    rec = json.loads((tmp_path / "registry.jsonl").read_text().splitlines()[0])
    for key in ("builder", "builder_kwargs", "feature_set", "train_row_filter", "cv_n_splits",
                "cv_n_repeats", "seed", "fold_scores", "cv_mean", "config_hash"):
        assert key in rec
    assert rec["seed"] == config.SEED and len(rec["fold_scores"]) == 5
    report = evaluate.run_registered("E901", verbose=False)
    assert report["reproduces_registered_scores"] is True
    assert report["reproduced_mean"] == pytest.approx(exp.cv_mean, abs=2e-4)


# ---------------------------------------------------------------------------
# 7. Re-verification of director fixes F4 / F5
# ---------------------------------------------------------------------------


def test_f5_exterior2nd_key_is_cemntbd():
    m = config.DICTIONARY_TO_DATA_VALUES["Exterior2nd"]
    assert m["CemntBd"] == "CmentBd" and "CmentBd" not in m


def test_f4_holdout_file_validated(tmp_path, monkeypatch):
    train = pd.read_csv(config.TRAIN_CSV)
    p = tmp_path / "h.json"
    monkeypatch.setattr(config, "HOLDOUT_IDS_PATH", p)
    ids = validation.load_or_create_holdout_ids(train)
    assert len(ids) == 219 and validation.load_or_create_holdout_ids(train) == ids
    p.write_text(json.dumps({"holdout_ids": ids[:-1] + [999999]}))
    with pytest.raises(ValueError):
        validation.load_or_create_holdout_ids(train)
    p.write_text(json.dumps({"holdout_ids": ids[:-1] + [ids[0]]}))
    with pytest.raises(ValueError):
        validation.load_or_create_holdout_ids(train)


def test_dry_run_results_must_not_be_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    common = _load_common()
    common.set_dry_run(120)
    try:
        with pytest.raises(Exception):
            common.run_and_register(
                "E902", title="t", hypothesis="h", rationale="r", implementation="i",
                builder="houseprices.models:build_mean_baseline", builder_kwargs={},
                interpret=lambda r: ("ok", "retain"),
            )
    finally:
        common.set_dry_run(None)


# ---------------------------------------------------------------------------
# Optional preprocessing components added by Agent A during the audit (default OFF, not used by E001-E004)
# ---------------------------------------------------------------------------


def test_default_preprocessor_has_no_optional_components(dev):
    X = dev.X.drop(columns="Id")
    r = repr(preprocess.make_preprocessor(X).get_params()["transformers"])
    assert "SkewLog1pTransformer" not in r and "KNNLotFrontageImputer" not in r
    assert "GroupMedianImputer" in r


def test_skew_transformer_learns_columns_from_fit_rows_only():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.exponential(size=300), "y": rng.normal(size=300) + 10})
    b = pd.DataFrame({"x": rng.normal(size=300) + 10, "y": rng.exponential(size=300)})
    ta, tb = preprocess.SkewLog1pTransformer().fit(a), preprocess.SkewLog1pTransformer().fit(b)
    assert ta.skewed_columns_ == ["x"] and tb.skewed_columns_ == ["y"]
    before = ta.log_mask_.copy()
    ta.transform(b)
    assert (ta.log_mask_ == before).all()


def test_knn_lotfrontage_fills_from_training_rows_only(dev):
    X = dev.X
    cols = ["LotFrontage", *preprocess.KNN_LOTFRONTAGE_PREDICTORS]
    tr, va = X.iloc[:700][cols], X.iloc[700:][cols]
    imp = preprocess.KNNLotFrontageImputer().fit(tr)
    snapshot = imp.knn_._fit_X.copy()
    imp.transform(va)
    np.testing.assert_array_equal(imp.knn_._fit_X, snapshot)  # new rows never join the donor pool
    assert np.isfinite(imp.transform(va)).all()


# ---------------------------------------------------------------------------
# Round 2 addendum: fitted transformers in preprocess.py and Agent A's experiment runners
# ---------------------------------------------------------------------------

KNN_COLS = ["LotFrontage", *preprocess.KNN_LOTFRONTAGE_PREDICTORS]


def test_knn_transform_is_row_independent_and_ignores_new_lotfrontage_values(dev):
    X = dev.X
    imp = preprocess.KNNLotFrontageImputer().fit(X.iloc[:700][KNN_COLS])
    va = X.iloc[700:760][KNN_COLS].copy()
    batch = imp.transform(va).ravel()
    single = np.array([imp.transform(va.iloc[[i]])[0, 0] for i in range(len(va))])
    np.testing.assert_allclose(batch, single)
    # Changing LotFrontage of OTHER validation rows must not move a row's imputed value:
    # validation rows never become donors.
    miss = va.index[va["LotFrontage"].isna()]
    assert len(miss) > 0
    changed = va.copy()
    others = changed.index.difference(miss)
    changed.loc[others, "LotFrontage"] = 999.0
    np.testing.assert_allclose(imp.transform(changed).ravel()[va.index.get_indexer(miss)],
                               batch[va.index.get_indexer(miss)])
    # observed values are returned unchanged
    obs = va.index.get_indexer(others)
    np.testing.assert_allclose(batch[obs], va.loc[others, "LotFrontage"].to_numpy())


def test_knn_handles_nan_predictors_and_depends_on_fold(dev):
    X = dev.X
    a = preprocess.KNNLotFrontageImputer().fit(X.iloc[:500][KNN_COLS])
    b = preprocess.KNNLotFrontageImputer().fit(X.iloc[500:1000][KNN_COLS])
    probe = X.iloc[1000:1040][KNN_COLS].copy()
    probe["LotFrontage"] = np.nan
    probe.iloc[0, probe.columns.get_loc("LotArea")] = np.nan
    pa, pb = a.transform(probe), b.transform(probe)
    assert np.isfinite(pa).all() and np.isfinite(pb).all()
    assert not np.allclose(pa, pb)  # neighbour statistics come from each fit sample


def test_make_preprocessor_step_order_knn_scale_and_skew(dev):
    X = dev.X.drop(columns="Id")
    ct = preprocess.make_preprocessor(X, lotfrontage="knn", skew_log1p=True, scale=True)
    named = {n: t for n, t, _ in ct.transformers}
    assert [s for s, _ in named["lotfrontage"].steps] == ["knn", "scale"]      # scaling after the fill
    assert [s for s, _ in named["numeric"].steps] == ["skew", "impute", "scale"]
    cols = {n: c for n, _, c in ct.transformers}
    assert cols["lotfrontage"] == KNN_COLS and "LotFrontage" not in cols["numeric"]


def test_skew_pipeline_in_cv_selects_columns_per_fold_and_is_row_independent(dev):
    X, y = dev.X, dev.y_log
    est = models.build_ridge(preprocessor_kwargs={"skew_log1p": True, "lotfrontage": "knn"})
    cv = KFold(5, shuffle=True, random_state=42)
    res = sk_cross_validate(est, X, y, cv=cv, return_estimator=True, scoring="neg_root_mean_squared_error")
    chosen = []
    for (tr, va), fitted in zip(cv.split(X), res["estimator"]):
        ct = fitted.named_steps["prep"].column_transformer_
        sk = ct.named_transformers_["numeric"].named_steps["skew"]
        num_cols = ct.transformers_[1][2]
        expected = [c for c in num_cols if X.iloc[tr][c].skew() > 0.75 and X.iloc[tr][c].min() >= 0]
        assert sk.skewed_columns_ == expected
        chosen.append(tuple(sk.skewed_columns_))
        pre = fitted.named_steps["prep"]
        whole = pre.transform(X.iloc[va[:20]])
        singles = np.vstack([pre.transform(X.iloc[[i]]) for i in va[:20]])
        np.testing.assert_allclose(whole, singles)
    assert np.isfinite(res["test_score"]).all()


def _ablation_common():
    import sys
    d = str(config.PROJECT_ROOT / "experiments" / "feature_ablation")
    if d not in sys.path:
        sys.path.insert(0, d)
    import _ablation_common
    return _ablation_common


def test_ablation_scripts_never_touch_holdout_and_use_only_common_entry_points():
    base = config.PROJECT_ROOT / "experiments" / "feature_ablation"
    for name in ("_ablation_common.py", "run_ablation.py", "run_lotfrontage.py"):
        src = (base / name).read_text(encoding="utf-8")
        assert "X_holdout" not in src and "y_holdout" not in src and "split_development_holdout" not in src
        assert "load_development_data" not in src and "cross_validate" not in src
        assert "holdout_ids" not in src
    common_src = (base / "_ablation_common.py").read_text(encoding="utf-8")
    assert "_common.run_cv" in common_src and "_common.run_and_register" in common_src
    assert "_common.find_registered" in common_src


def test_ablation_runner_skips_duplicate_hash_and_dry_run_writes_nothing(tmp_path, tmp_holdout, monkeypatch):
    ac = _ablation_common()
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    reg = tmp_path / "registry.jsonl"
    kw = dict(title="t", hypothesis="h", rationale="r", implementation="i",
              interpret=lambda r: ("ok", "retain"), tags=["x"])
    builder, kwargs = ac.model_spec("ridge", {"lotfrontage": "global_median"})
    try:
        dry = ac.Runner(dry_run=True, budget_minutes=1)
        dry.get(("base", "ridge", "g"), "base", builder, kwargs, **kw)
        assert not reg.exists()
    finally:
        ac._common.set_dry_run(None)
    real = ac.Runner(dry_run=False, budget_minutes=5)
    first = real.get(("base", "ridge", "g"), "base", builder, kwargs, **kw)
    again = ac.Runner(dry_run=False, budget_minutes=5).get(("base", "ridge", "g"), "base", builder, kwargs, **kw)
    assert first["id"] == again["id"]
    assert len(reg.read_text().splitlines()) == 1
    assert again["folds"] == pytest.approx(first["folds"], abs=1e-5)


def test_paired_stats_matches_hand_computation():
    ac = _ablation_common()
    d, se = ac.paired_stats([0.10, 0.12, 0.14], [0.11, 0.11, 0.11])
    assert d == pytest.approx(0.01)
    assert se == pytest.approx(np.std([-0.01, 0.01, 0.03], ddof=1) / np.sqrt(3))


# ---------------------------------------------------------------------------
# Round 3: ensemble.py (nested stacking and blending)
# ---------------------------------------------------------------------------

from sklearn.datasets import make_regression  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from houseprices import ensemble  # noqa: E402
from tests.test_ensemble import Memoriser  # noqa: E402,F401  (reuse the memoriser from Agent C's tests)


@pytest.fixture(scope="module")
def synth():
    X, y = make_regression(n_samples=300, n_features=8, n_informative=5, noise=10.0, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


class _Recorder(Ridge):
    """Ridge that records the row labels of every frame it is fit on (class-level log)."""

    log: list = []

    def fit(self, X, y, **kw):
        type(self).log.append(set(X.index))
        return super().fit(X, y, **kw)


class _LeakyStack(ensemble.NestedStackingRegressor):
    """MUTANT: meta-learner trained on in-sample base predictions (the classic leak)."""

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self.fitted_bases_ = ensemble._fit_all(self.base_estimators, X, y_arr)
        self.meta_ = Ridge(alpha=1e-6).fit(ensemble._predict_matrix(self.fitted_bases_, X), y_arr)
        return self


def test_nested_stack_outer_validation_rows_never_touch_any_fit(synth):
    X, y = synth
    _Recorder.log = []
    stack = ensemble.NestedStackingRegressor([("rec", _Recorder())], Ridge())
    cv = KFold(5, shuffle=True, random_state=42)
    res = sk_cross_validate(stack, X, y, cv=cv, return_estimator=True, scoring="neg_root_mean_squared_error")
    assert len(_Recorder.log) == 5 * (5 + 1)               # 5 inner fits + 1 refit per outer fold
    for k, ((tr, va), fitted) in enumerate(zip(cv.split(X), res["estimator"])):
        va_labels, tr_labels = set(X.index[va]), set(X.index[tr])
        for fit_rows in _Recorder.log[k * 6:(k + 1) * 6]:
            assert not fit_rows & va_labels and fit_rows <= tr_labels
        assert tr_labels.isdisjoint(va_labels)
        assert len(fitted.fitted_bases_) == 1 and fitted.inner_oof_.shape == (len(tr), 1)


def test_nested_stack_meta_is_fit_on_inner_oof_of_outer_train_only(synth):
    X, y = synth
    bases_ = [("a", Ridge(alpha=1.0)), ("b", Ridge(alpha=1000.0))]
    cv = KFold(5, shuffle=True, random_state=42)
    stack = ensemble.NestedStackingRegressor(bases_, Ridge(alpha=1.0))
    res = sk_cross_validate(stack, X, y, cv=cv, return_estimator=True)
    for (tr, va), fitted in zip(cv.split(X), res["estimator"]):
        oof = ensemble.inner_oof_matrix(bases_, X.iloc[tr], y.iloc[tr].to_numpy(), 5, 42)
        np.testing.assert_allclose(fitted.inner_oof_, oof)
        np.testing.assert_allclose(fitted.meta_.coef_, Ridge(alpha=1.0).fit(oof, y.iloc[tr]).coef_)
        # base refit on the FULL outer-train partition
        ref = Ridge(alpha=1.0).fit(X.iloc[tr], y.iloc[tr])
        np.testing.assert_allclose(fitted.fitted_bases_[0][1].coef_, ref.coef_)


def test_leaky_mutant_is_caught_by_a_two_base_memoriser_probe(synth):
    """A memoriser plus an honest Ridge: nested stack should lean on Ridge; the leaky mutant leans on the memoriser."""
    X, y = synth
    bases_ = [("mem", Memoriser()), ("ridge", Ridge(alpha=1.0))]
    honest = validation.cross_validate(ensemble.NestedStackingRegressor(bases_, Ridge(alpha=1e-6)), X, y).mean
    leaky = validation.cross_validate(_LeakyStack(bases_, Ridge()), X, y).mean
    ridge_only = validation.cross_validate(Ridge(alpha=1.0), X, y).mean
    assert honest == pytest.approx(ridge_only, rel=0.15)
    assert leaky > 3 * honest


def test_single_memoriser_test_cannot_tell_honest_from_leaky_stack(synth):
    """Documents the sensitivity limit of test_ensemble.py's memoriser test (finding F16)."""
    X, y = synth
    mem = [("mem", Memoriser())]
    honest = validation.cross_validate(ensemble.NestedStackingRegressor(mem, Ridge(alpha=1e-6)), X, y).mean
    leaky = validation.cross_validate(_LeakyStack(mem, Ridge()), X, y).mean
    assert honest > 0.8 * float(np.std(y)) and leaky > 0.8 * float(np.std(y))   # both pass the existing assertion


def test_blend_weights_none_are_learned_per_outer_fold_from_outer_train_only(synth):
    X, y = synth
    from sklearn.neighbors import KNeighborsRegressor
    bases_ = [("a", Ridge(alpha=1.0)), ("b", KNeighborsRegressor(n_neighbors=7))]
    cv = KFold(5, shuffle=True, random_state=42)
    res = sk_cross_validate(ensemble.BlendingRegressor(bases_), X, y, cv=cv, return_estimator=True)
    all_w = []
    for (tr, _), fitted in zip(cv.split(X), res["estimator"]):
        oof = ensemble.inner_oof_matrix(bases_, X.iloc[tr], y.iloc[tr].to_numpy(), 5, 42)
        np.testing.assert_allclose(fitted.weights_, ensemble.fit_blend_weights(oof, y.iloc[tr].to_numpy()))
        all_w.append(fitted.weights_)
    assert all(abs(w.sum() - 1) < 1e-9 for w in all_w)
    # fixed weights: nothing learned, template untouched
    fixed = ensemble.BlendingRegressor(bases_, weights=[0.5, 0.5])
    assert fixed.fit(X, y).weights_.tolist() == [0.5, 0.5]


def test_optimistic_diagnostics_cannot_flow_through_run_and_register(tmp_path, tmp_holdout, monkeypatch):
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    common = _load_common()
    score = ensemble.optimistic_oof_stack_score([("r", Ridge())], Ridge(), pd.DataFrame(np.random.rand(60, 2)),
                                               pd.Series(np.random.rand(60)))
    assert isinstance(score, float)
    with pytest.raises(AttributeError):
        common.run_and_register("E903", title="t", hypothesis="h", rationale="r", implementation="i",
                                builder="houseprices.ensemble:build_stack", builder_kwargs={},
                                interpret=lambda r: ("x", "retain"), result=score)
    assert not (tmp_path / "registry.jsonl").exists()


def test_stack_registers_and_reproduces_through_evaluate(tmp_path, tmp_holdout, monkeypatch):
    from houseprices import evaluate

    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    common = _load_common()
    kw = {"base_builders": ["houseprices.models:build_ridge", "houseprices.models:build_ridge"],
          "base_kwargs": [{"alpha": 3.0}, {"alpha": 30.0}], "meta": "ridge"}
    common.run_and_register("E904", title="t", hypothesis="h", rationale="r", implementation="i",
                            builder="houseprices.ensemble:build_stack", builder_kwargs=kw,
                            interpret=lambda r: ("ok", "retain"))
    assert evaluate.run_registered("E904", verbose=False)["reproduces_registered_scores"] is True


# ---------------------------------------------------------------------------
# Round 4: tuning / grid / outlier scripts and the holdout guard
# ---------------------------------------------------------------------------

import sys as _sys  # noqa: E402
import warnings as _warnings  # noqa: E402


def _import_script(subdir: str, name: str):
    d = str(config.PROJECT_ROOT / "experiments" / subdir)
    if d not in _sys.path:
        _sys.path.insert(0, d)
    _sys.path.insert(0, str(config.PROJECT_ROOT / "experiments" / "models"))
    return importlib.import_module(name)


def test_round4_scripts_never_reference_holdout_or_bypass_run_cv():
    for sub, name in (("models", "run_outlier_policy.py"), ("models", "run_linear_family.py"),
                      ("tuning", "run_optuna_stage1.py")):
        src = (config.PROJECT_ROOT / "experiments" / sub / name).read_text(encoding="utf-8")
        for bad in ("X_holdout", "y_holdout", "score_holdout_once", "holdout_ids", "split_development_holdout",
                    "load_development_data", "validation.cross_validate", "make_cv"):
            assert bad not in src, (name, bad)
        assert "_common.run_cv" in src


def _stub_cv(*a, **k):
    return validation.CVResult([0.1] * 5, np.array([]), 0.0)


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_optuna_registered_kwargs_rebuild_identical_pipeline(name):
    optuna = pytest.importorskip("optuna")
    mod = _import_script("tuning", "run_optuna_stage1")
    seen = {}

    def fake_run_cv(builder, kwargs, fs, flt):
        seen.update(builder=builder, kwargs=kwargs, fs=fs, flt=flt)
        return _stub_cv()

    mp = pytest.MonkeyPatch()
    mp.setattr(mod._common, "run_cv", fake_run_cv)
    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(mod.make_objective(name, {}, "base", "keep_all", False), n_trials=1)
    finally:
        mp.undo()
    trial = study.trials[0]
    assert seen["fs"] == "base" and seen["flt"] == "keep_all" and seen["builder"] == mod.BUILDERS[name]
    recorded = json.loads(trial.user_attrs["builder_kwargs"])
    assert recorded == seen["kwargs"]                       # registered dict == dict that was scored
    # the space output equals the recorded dict (nothing rebuilt from trial.params)
    space = {}
    probe = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    probe.optimize(lambda t: (space.update(models.OPTUNA_SPACES[name](t)), 0.0)[1], n_trials=1)
    assert space == recorded
    if name == "lightgbm":
        assert recorded["subsample_freq"] == 1 and "subsample_freq" not in trial.params
    builder = data.resolve_callable(mod.BUILDERS[name])
    direct, rebuilt = builder(**seen["kwargs"]), builder(**json.loads(json.dumps(recorded)))
    assert repr(direct) == repr(rebuilt)
    assert repr(direct.named_steps["model"].get_params()) == repr(rebuilt.named_steps["model"].get_params())
    from houseprices import registry as _reg

    def h(kw):
        return _reg.Experiment(id="", agent="a", title="", hypothesis="", rationale="", implementation="",
                               evaluation="", builder=mod.BUILDERS[name], builder_kwargs=kw).config_hash

    assert h(recorded) == h(json.loads(json.dumps(recorded)))


def test_optuna_catboost_native_extra_is_in_registered_kwargs():
    optuna = pytest.importorskip("optuna")
    mod = _import_script("tuning", "run_optuna_stage1")
    mp = pytest.MonkeyPatch()
    mp.setattr(mod._common, "run_cv", _stub_cv)
    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        study.optimize(mod.make_objective("catboost", {"native_categoricals": True}, "base", "keep_all", False),
                       n_trials=1)
    finally:
        mp.undo()
    assert json.loads(study.trials[0].user_attrs["builder_kwargs"])["native_categoricals"] is True


def test_stage2_rule_matches_director_rule():
    mod = _import_script("tuning", "run_optuna_stage1")
    flat = [0.14] * 25
    assert mod.stage2_rule(flat, reference=0.1369)["stage2_recommended"] is False   # gap 0.0031 > 0.003, flat
    assert mod.stage2_rule(flat, reference=0.1371)["stage2_recommended"] is True    # gap 0.0029 <= 0.003
    improving = [0.20] * 14 + list(np.linspace(0.19, 0.15, 11))
    r = mod.stage2_rule(improving, reference=0.10)
    assert r["improving_last_10"] and r["stage2_recommended"] and not r["within_gap"]
    early = [0.15] * 25                                                              # nothing new in the last window
    assert mod.stage2_rule(early, reference=0.10)["improving_last_10"] is False
    assert (mod.STAGE2_GAP, mod.STAGE2_WINDOW, mod.TUNING_CAP_HOURS) == (0.003, 10, 3.0)


def test_linear_grid_scores_only_through_run_cv_with_skew_and_counts_warnings():
    from sklearn.exceptions import ConvergenceWarning
    mod = _import_script("models", "run_linear_family")
    calls = []

    def fake_run_cv(builder, kwargs, fs, flt):
        calls.append((builder, dict(kwargs), fs, flt))
        if len(calls) == 3:
            _warnings.warn("x", ConvergenceWarning)
        return validation.CVResult([0.1, 0.11, 0.12, 0.13, 0.14], np.array([]), 0.0)

    mp = pytest.MonkeyPatch()
    mp.setattr(mod._common, "run_cv", fake_run_cv)
    try:
        grid = mod.run_grid("base", "keep_all")
    finally:
        mp.undo()
    assert len(grid) == len(calls) == 2 * (15 + 13 + 40) == 136
    assert all(c[1]["preprocessor_kwargs"] == {"skew_log1p": True} for c in calls)
    assert all(c[1]["scaler"] in ("robust", "standard") and c[2] == "base" and c[3] == "keep_all" for c in calls)
    assert int(grid["convergence_warnings"].sum()) == 1


def test_linear_best_registers_complete_kwargs_and_only_three_experiments(tmp_path, tmp_holdout, monkeypatch):
    mod = _import_script("models", "run_linear_family")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    rng = np.random.default_rng(0)
    monkeypatch.setattr(mod._common, "run_cv", lambda b, k, fs, flt: validation.CVResult(
        list(0.13 + 0.01 * rng.random(5)), np.array([]), 0.1))
    grid = mod.run_grid("base", "keep_all", thin=5)
    for fam, (builder, _) in mod.FAMILIES.items():
        mod.register_best(fam, builder, grid[grid["model"] == fam], "base", "keep_all", "grid.csv")
    recs = [json.loads(line) for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert len(recs) == 3
    for r in recs:
        assert r["builder_kwargs"]["preprocessor_kwargs"] == {"skew_log1p": True} and "scaler" in r["builder_kwargs"]
        est = data.resolve_callable(r["builder"])(**r["builder_kwargs"])      # rebuildable
        assert "model" in est.named_steps
        assert "ConvergenceWarnings" in r["notes"]


def test_repeated_kfold_finalist_check_shares_its_first_repeat_with_selection_folds(dev):
    """F20: make_cv(5, 3, 42) repeat 1 is exactly the KFold(5, shuffle, 42) split used to SELECT configs."""
    sel = [tuple(v) for _, v in validation.make_cv(5, 1, 42).split(dev.X)]
    rep = [tuple(v) for _, v in list(validation.make_cv(5, 3, 42).split(dev.X))[:5]]
    assert sel == rep
    other = [tuple(v) for _, v in list(validation.make_cv(5, 3, 7).split(dev.X))[:5]]
    assert sel != other


# --- holdout guard -----------------------------------------------------------------------------


@pytest.fixture()
def usage_log(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "HOLDOUT_USAGE_LOG", tmp_path / "holdout_usage.log")
    return tmp_path / "holdout_usage.log"


def _fitted_ridge(dev):
    return models.build_ridge(alpha=10.0).fit(dev.X, dev.y_log)


def test_holdout_guard_single_use_logged_forced_marked(dev, usage_log):
    est = _fitted_ridge(dev)
    ev = validation.score_holdout_once(est, dev.X_holdout, dev.y_holdout_log, "E999", note="probe")
    assert ev["forced"] is False and ev["n_previous_uses"] == 0 and ev["n_holdout"] == len(dev.X_holdout)
    assert 0.05 < ev["rmse_log"] < 0.3
    assert len(validation.holdout_usage()) == 1
    with pytest.raises(validation.HoldoutAlreadyUsedError):
        validation.score_holdout_once(est, dev.X_holdout, dev.y_holdout_log, "E999")
    assert len(validation.holdout_usage()) == 1            # a refused call logs nothing
    forced = validation.score_holdout_once(est, dev.X_holdout, dev.y_holdout_log, "E999", force=True)
    assert forced["forced"] is True and forced["n_previous_uses"] == 1
    assert [e["forced"] for e in validation.holdout_usage()] == [False, True]


def test_holdout_guard_nonfinite_prediction_does_not_consume_the_single_use(dev, usage_log):
    class Bad:
        def predict(self, X):
            return np.full(len(X), np.nan)

    with pytest.raises(ValueError):
        validation.score_holdout_once(Bad(), dev.X_holdout, dev.y_holdout_log, "E999")
    assert validation.holdout_usage() == []


def test_holdout_is_disjoint_from_development_and_has_219_rows(dev):
    assert set(dev.X_holdout["Id"]).isdisjoint(set(dev.X["Id"]))
    assert len(dev.X_holdout) == 219


@pytest.mark.xfail(strict=True, reason="F21: score_holdout_once accepts any frame, so a mistaken call with development rows consumes the single use")
def test_holdout_guard_rejects_non_holdout_frame(dev, usage_log):
    est = _fitted_ridge(dev)
    with pytest.raises(ValueError):
        validation.score_holdout_once(est, dev.X.iloc[:219], dev.y_log.iloc[:219], "E999")
