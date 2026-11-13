"""C8: re-score the finalist configurations on folds they were NOT selected on.

    python experiments/finalists/run_finalists.py --ids E056,E060,E061 [--repeats 3] [--seed 7]

For each registered experiment the builder is rebuilt from the registry and cross-validated
with RepeatedKFold(5 x repeats, seed = config.CV_SEED_FINALISTS). Decision D17 / audit F20:
the selection folds (seed 42) carry about +0.002-0.003 selection optimism, so the finalist
estimate must come from a different partition. Each result is registered as a NEW experiment
(agent "director", tags ["finalist"], cv_n_repeats = repeats, seed = finalist seed) so that
``python -m houseprices.evaluate --experiment <new id>`` reproduces it, and a comparison table
is written to experiments/finalists/finalists_<timestamp>.csv.

Paired comparison: all finalists are scored on the same 15 folds, so the table also reports
the per-fold paired difference against the best finalist and its standard error (sd / sqrt(15)).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from _common import find_registered  # noqa: E402

from houseprices import config  # noqa: E402
from houseprices import data as data_mod  # noqa: E402
from houseprices import registry, validation  # noqa: E402

OUT_DIR = config.EXPERIMENTS_DIR / "finalists"


def score_finalist(rec: dict, repeats: int, seed: int) -> validation.CVResult:
    kwargs = rec.get("builder_kwargs") or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    estimator = data_mod.resolve_callable(rec["builder"])(**kwargs)
    dev = data_mod.load_development_data(rec.get("feature_set", "base"))
    cv = validation.make_cv(int(rec.get("cv_n_splits", 5)), repeats, seed)
    row_filter = data_mod.get_train_row_filter(rec.get("train_row_filter", "keep_all"))
    return validation.cross_validate(estimator, dev.X, dev.y_log, cv, row_filter)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", required=True, help="comma-separated registry ids of the finalists")
    parser.add_argument("--repeats", type=int, default=config.CV_N_REPEATS_FINALISTS)
    parser.add_argument("--seed", type=int, default=config.CV_SEED_FINALISTS)
    args = parser.parse_args(argv)
    if args.seed == config.SEED:
        raise SystemExit("finalist seed must differ from the selection seed (D17)")

    rows = []
    fold_matrix = {}
    for exp_id in [i.strip() for i in args.ids.split(",") if i.strip()]:
        rec = registry.get_experiment(exp_id)
        kwargs = rec.get("builder_kwargs") or {}
        if isinstance(kwargs, str):
            kwargs = json.loads(kwargs)
        existing = find_registered(rec["builder"], kwargs, rec.get("feature_set", "base"),
                                   rec.get("train_row_filter", "keep_all"), n_repeats=args.repeats,
                                   seed=args.seed)
        if existing is not None:
            prev = registry.get_experiment(existing)
            print(f"{exp_id}: finalist check already registered as {existing} ({prev['cv_mean']:.5f}); reusing")
            scores = [float(s) for s in prev["fold_scores"]]
            new_id = existing
            seconds = prev.get("fit_seconds")
        else:
            t0 = time.perf_counter()
            res = score_finalist(rec, args.repeats, args.seed)
            seconds = round(time.perf_counter() - t0, 1)
            scores = res.fold_scores
            exp = registry.Experiment(
                id="", agent="director",
                title=f"Finalist check: {rec['title']} (RepeatedKFold 5x{args.repeats}, seed {args.seed})",
                hypothesis="The selection-fold score of this configuration is optimistic by about 0.002-0.003; "
                           "fresh folds give an unbiased estimate of the chosen configuration.",
                rationale="Decision D17 / audit F20: finalists must be compared on folds disjoint from the selection partition.",
                implementation=f"Same builder and kwargs as {exp_id}; only the CV splitter changes.",
                evaluation=f"RepeatedKFold(5x{args.repeats}, seed {args.seed}) on log1p(SalePrice); RMSE on log scale; preprocessing refit per fold",
                builder=rec["builder"], builder_kwargs=kwargs, feature_set=rec.get("feature_set", "base"),
                train_row_filter=rec.get("train_row_filter", "keep_all"), cv_n_splits=int(rec.get("cv_n_splits", 5)),
                cv_n_repeats=args.repeats, seed=args.seed,
                fold_scores=[round(s, 6) for s in scores], cv_mean=round(float(np.mean(scores)), 6),
                cv_std=round(float(np.std(scores, ddof=1)), 6), fit_seconds=seconds,
                interpretation=f"Selection-fold score {rec['cv_mean']:.5f} (seed 42) vs fresh-fold score "
                               f"{np.mean(scores):.5f} (seed {args.seed}); difference {np.mean(scores) - rec['cv_mean']:+.5f}.",
                decision="pending", tags=["finalist", *[t for t in (rec.get("tags") or []) if t != "finalist"]],
                notes=f"source experiment {exp_id}",
            )
            registry.append(exp)
            new_id = exp.id
            print(f"{exp_id} -> {new_id}: {exp.cv_mean:.5f} +/- {exp.cv_std:.4f} ({seconds}s)")
        fold_matrix[exp_id] = np.asarray(scores)
        rows.append({"source_id": exp_id, "finalist_id": new_id, "title": rec["title"],
                     "selection_cv_mean": rec["cv_mean"], "fresh_cv_mean": round(float(np.mean(scores)), 5),
                     "fresh_cv_std": round(float(np.std(scores, ddof=1)), 5),
                     "optimism": round(float(np.mean(scores) - rec["cv_mean"]), 5), "seconds": seconds})

    table = pd.DataFrame(rows).sort_values("fresh_cv_mean").reset_index(drop=True)
    best = table.loc[0, "source_id"]
    diffs = []
    for sid in table["source_id"]:
        d = fold_matrix[sid] - fold_matrix[best]
        diffs.append((round(float(d.mean()), 5), round(float(d.std(ddof=1) / np.sqrt(len(d))), 5) if len(d) > 1 else 0.0))
    table["paired_diff_vs_best"] = [d[0] for d in diffs]
    table["paired_se"] = [d[1] for d in diffs]
    table["within_one_se_of_best"] = table["paired_diff_vs_best"] <= table["paired_se"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"finalists_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
