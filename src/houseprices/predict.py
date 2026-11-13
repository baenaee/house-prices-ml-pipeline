"""Load a fitted artefact, predict the competition test set and write a submission file.

    python -m houseprices.predict --model outputs/models/E060_all.joblib
    python -m houseprices.predict --model outputs/models/E060_all.joblib --out outputs/submissions/catboost_stage1.csv

The file is written for the user to upload by hand; nothing here talks to Kaggle.
Checks performed before writing: predictions finite and positive after ``expm1``; exactly the
Ids of ``sample_submission.csv`` in the same order; columns ``Id, SalePrice``. The predictions
for the three test-only anomalies (Ids 2593, 2550, 1877; decision D18) are printed for a
manual sanity check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from houseprices import config
from houseprices import data as data_mod
from houseprices import validation

ANOMALY_TEST_IDS = [2593, 2550, 1877]


def predict_test(model_path: Path, out_path: Path | None = None, verbose: bool = True) -> tuple[Path, dict]:
    model_path = Path(model_path)
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    estimator = joblib.load(model_path)

    dev = data_mod.load_development_data(meta["feature_set"], with_holdout_split=False)
    X_test = dev.X_test
    pred_log = np.asarray(estimator.predict(X_test), dtype=float)
    if not np.all(np.isfinite(pred_log)):
        raise ValueError(f"{int((~np.isfinite(pred_log)).sum())} non-finite log predictions")
    price = validation.from_log_target(pred_log)
    if not np.all(np.isfinite(price)):
        raise ValueError("non-finite price predictions after expm1 (log predictions too large)")
    if not np.all(price > 0):
        raise ValueError("non-positive price predictions")

    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({config.ID_COL: X_test[config.ID_COL].to_numpy(), config.TARGET: price})
    sub = sample[[config.ID_COL]].merge(sub, on=config.ID_COL, how="left")
    if sub[config.TARGET].isna().any() or len(sub) != len(sample):
        raise ValueError("submission Ids do not match sample_submission.csv")
    if not (sub[config.ID_COL].to_numpy() == sample[config.ID_COL].to_numpy()).all():
        raise ValueError("submission Id order differs from sample_submission.csv")

    out_path = Path(out_path) if out_path is not None else config.SUBMISSIONS_DIR / f"{model_path.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    anomalies = sub.loc[sub[config.ID_COL].isin(ANOMALY_TEST_IDS)].set_index(config.ID_COL)[config.TARGET].round(0).to_dict()
    report = {
        "model": str(model_path),
        "experiment_id": meta.get("experiment_id"),
        "rows_fit": meta.get("rows"),
        "n_predictions": int(len(sub)),
        "price_min": round(float(price.min()), 0),
        "price_median": round(float(np.median(price)), 0),
        "price_max": round(float(price.max()), 0),
        "log_pred_mean": round(float(pred_log.mean()), 4),
        "anomaly_test_ids_prediction": {int(k): v for k, v in anomalies.items()},
        "submission": str(out_path),
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return out_path, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to a .joblib written by houseprices.train")
    parser.add_argument("--out", default=None, help="submission CSV path (default outputs/submissions/<model>.csv)")
    args = parser.parse_args(argv)
    predict_test(Path(args.model), Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
