"""C8: the single evaluation of the 219-row holdout for the FROZEN procedure (ruling D22).

    python experiments/finalists/score_holdout.py --experiment E066 --model outputs/models/E066_development.joblib --note "D22"

Run exactly once, after the freeze and the auditor's approval. The score goes through
``validation.score_holdout_once``, which logs the use in ``experiments/holdout_usage.log``,
refuses any frame that is not the registered holdout and refuses a second call.

The model must be the artefact fitted on the DEVELOPMENT rows (``train --rows development``);
the sidecar is checked. The predictions made by this single evaluation are saved next to the
log (Id, actual, predicted, residual on the log scale) so the report can describe the same
evaluation in more detail without predicting on the holdout again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from houseprices import config, registry, validation
from houseprices import data as data_mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="frozen registry id")
    ap.add_argument("--model", required=True, help="joblib fitted on development rows by houseprices.train")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    if validation.holdout_usage():
        raise SystemExit(f"holdout already scored: {validation.holdout_usage()[0]}")
    model_path = Path(args.model)
    meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    rec = registry.get_experiment(args.experiment)
    if meta["experiment_id"] != args.experiment or meta["config_hash"] != rec["config_hash"]:
        raise SystemExit("artefact does not belong to the frozen experiment")
    if meta["rows"] != "development" or meta["includes_holdout_rows"] or meta["override"]:
        raise SystemExit("artefact must be the un-overridden development-rows fit")

    dev = data_mod.load_development_data(rec["feature_set"])
    model = joblib.load(model_path)
    event = validation.score_holdout_once(model, dev.X_holdout, dev.y_holdout_log, args.experiment, note=args.note)
    print(json.dumps(event, indent=2))

    # Same single evaluation, kept for the report (deterministic re-prediction of the logged call).
    pred = np.asarray(model.predict(dev.X_holdout), dtype=float)
    assert abs(validation.rmse_log(dev.y_holdout_log, pred) - event["rmse_log"]) < 1e-6
    out = pd.DataFrame({
        config.ID_COL: dev.X_holdout[config.ID_COL].to_numpy(),
        "y_log": dev.y_holdout_log.to_numpy(),
        "pred_log": pred,
        "residual_log": dev.y_holdout_log.to_numpy() - pred,
    })
    out_path = config.EXPERIMENTS_DIR / "finalists" / f"holdout_predictions_{args.experiment}.csv"
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
