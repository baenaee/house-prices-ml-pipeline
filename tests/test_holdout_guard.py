"""Director-owned tests: the holdout can be scored exactly once (binding correction 3)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from houseprices import validation


@pytest.fixture
def redirected_log(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "HOLDOUT_USAGE_LOG", tmp_path / "holdout_usage.log")
    ids_path = tmp_path / "holdout_ids.json"
    ids_path.write_text(json.dumps({"holdout_ids": list(range(10))}), encoding="utf-8")
    monkeypatch.setattr(validation.config, "HOLDOUT_IDS_PATH", ids_path)
    return tmp_path / "holdout_usage.log"


def test_non_holdout_frame_is_refused_and_does_not_consume_the_use(redirected_log):
    X, y = _frames()
    est = DummyRegressor(strategy="mean").fit(X, y)
    other = X.assign(Id=X["Id"] + 100)          # development-like rows with foreign ids
    with pytest.raises(ValueError, match="not the registered holdout"):
        validation.score_holdout_once(est, other, y, "E999")
    with pytest.raises(ValueError, match="not the registered holdout"):
        validation.score_holdout_once(est, X.iloc[:5], y.iloc[:5], "E999")   # subset of the holdout
    assert validation.holdout_usage() == []


def _frames():
    X = pd.DataFrame({"Id": range(10), "a": np.arange(10.0)})
    y = pd.Series(np.log1p(np.arange(10.0) * 1000 + 50000))
    return X, y


def test_first_use_is_logged_and_scored(redirected_log):
    X, y = _frames()
    est = DummyRegressor(strategy="mean").fit(X, y)
    event = validation.score_holdout_once(est, X, y, "E999", note="test")
    expected = validation.rmse_log(y, est.predict(X))
    assert event["rmse_log"] == round(expected, 6)
    assert event["forced"] is False and event["n_previous_uses"] == 0
    assert len(validation.holdout_usage()) == 1


def test_second_use_is_refused(redirected_log):
    X, y = _frames()
    est = DummyRegressor(strategy="mean").fit(X, y)
    validation.score_holdout_once(est, X, y, "E999")
    with pytest.raises(validation.HoldoutAlreadyUsedError):
        validation.score_holdout_once(est, X, y, "E999")
    assert len(validation.holdout_usage()) == 1


def test_forced_second_use_is_marked(redirected_log):
    X, y = _frames()
    est = DummyRegressor(strategy="mean").fit(X, y)
    validation.score_holdout_once(est, X, y, "E999")
    event = validation.score_holdout_once(est, X, y, "E999", force=True, note="documented reason")
    assert event["forced"] is True and event["n_previous_uses"] == 1
    assert len(validation.holdout_usage()) == 2


def test_real_log_is_still_empty():
    """No holdout evaluation has happened yet in this project (check before freezing)."""
    assert validation.holdout_usage() == []
