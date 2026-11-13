"""Director-owned end-to-end test: registry -> train -> predict -> valid submission file.

Uses the cheap registered Ridge baseline (E003) so the test runs in a few seconds. Artefacts go
to a temporary directory; nothing under outputs/ is touched.
"""

import json

import numpy as np
import pandas as pd
import pytest

from houseprices import config, predict, train


@pytest.fixture(scope="module")
def artefact(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("models")
    model_path, meta = train.fit_registered("E003", rows="development", out_dir=out_dir, verbose=False)
    return model_path, meta


def test_train_writes_model_and_sidecar(artefact):
    model_path, meta = artefact
    assert model_path.exists() and model_path.with_suffix(".json").exists()
    side = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["experiment_id"] == "E003" and side["rows"] == "development"
    assert side["n_rows_fit"] == 1460 - 219          # development rows only
    assert side["config_hash"] == meta["config_hash"]
    assert "evaluate" in side["note"]                  # the sidecar says it is not CV evidence


def test_override_is_merged_one_level_deep():
    merged = train.merge_override({"alpha": 1.0, "preprocessor_kwargs": {"skew_log1p": True}},
                                  {"preprocessor_kwargs": {"clip_years": True}, "alpha": 2.0})
    assert merged == {"alpha": 2.0, "preprocessor_kwargs": {"skew_log1p": True, "clip_years": True}}


def test_predict_writes_valid_submission(artefact, tmp_path):
    model_path, _ = artefact
    out = tmp_path / "sub.csv"
    path, report = predict.predict_test(model_path, out, verbose=False)
    sub = pd.read_csv(path)
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    assert list(sub.columns) == [config.ID_COL, config.TARGET]
    assert len(sub) == len(sample) == 1459
    assert (sub[config.ID_COL].to_numpy() == sample[config.ID_COL].to_numpy()).all()
    assert np.isfinite(sub[config.TARGET]).all() and (sub[config.TARGET] > 0).all()
    # plausible price range for Ames
    assert 30_000 < report["price_median"] < 300_000
    assert set(report["anomaly_test_ids_prediction"]) == {2593, 2550, 1877}


def test_fit_rows_all_uses_every_training_row(tmp_path):
    model_path, meta = train.fit_registered("E001", rows="all", out_dir=tmp_path, verbose=False)
    assert meta["n_rows_fit"] == 1460
