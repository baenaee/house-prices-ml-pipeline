"""Reproduce the cross-validation of a registered experiment.

    python -m houseprices.evaluate --experiment E012
    python -m houseprices.evaluate --experiment E012 --repeats 3   # stability check

This command is the ONLY accepted evidence that a reported CV score is reproducible.
``train.py`` and ``predict.py`` fit on all rows and produce a submission; they do not
reproduce CV numbers.

How it works: the registry stores a *builder* ("module:function") and its keyword
arguments. The builder returns a complete estimator pipeline. We rebuild it, load the
same development data (same cleaning, same holdout split, same feature groups), run the
same seeded folds with the same training-row filter, and compare fold scores.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from houseprices import data as data_mod
from houseprices import registry, validation


def run_registered(experiment_id: str, repeats: int | None = None, verbose: bool = True) -> dict:
    rec = registry.get_experiment(experiment_id)
    builder = data_mod.resolve_callable(rec["builder"])
    kwargs = rec.get("builder_kwargs") or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    estimator = builder(**kwargs)

    dev = data_mod.load_development_data(rec.get("feature_set", "base"))
    n_repeats = repeats if repeats is not None else int(rec.get("cv_n_repeats", 1))
    cv = validation.make_cv(int(rec.get("cv_n_splits", 5)), n_repeats, int(rec.get("seed", 42)))
    row_filter = data_mod.get_train_row_filter(rec.get("train_row_filter", "keep_all"))

    result = validation.cross_validate(estimator, dev.X, dev.y_log, cv, row_filter)
    registered = [float(s) for s in (rec.get("fold_scores") or [])]
    same_protocol = n_repeats == int(rec.get("cv_n_repeats", 1))
    match = (
        same_protocol
        and len(registered) == len(result.fold_scores)
        and np.allclose(registered, result.fold_scores, atol=2e-4)
    )
    report = {
        "experiment": experiment_id,
        "title": rec.get("title"),
        "builder": rec["builder"],
        "feature_set": rec.get("feature_set"),
        "train_row_filter": rec.get("train_row_filter"),
        "cv": f"{cv.__class__.__name__}(n_splits={rec.get('cv_n_splits', 5)}, n_repeats={n_repeats}, seed={rec.get('seed', 42)})",
        "registered_fold_scores": registered,
        "reproduced_fold_scores": [round(s, 5) for s in result.fold_scores],
        "reproduced_mean": round(result.mean, 5),
        "reproduced_std": round(result.std, 5),
        "registered_mean": rec.get("cv_mean"),
        "fit_seconds": round(result.fit_seconds, 1),
        "reproduces_registered_scores": bool(match) if same_protocol else "n/a (different repeats)",
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="registry id, e.g. E012")
    parser.add_argument("--repeats", type=int, default=None, help="override n_repeats (stability check)")
    args = parser.parse_args(argv)
    report = run_registered(args.experiment, args.repeats)
    ok = report["reproduces_registered_scores"]
    return 0 if ok in (True, "n/a (different repeats)") else 1


if __name__ == "__main__":
    sys.exit(main())
