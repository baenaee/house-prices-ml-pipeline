"""C1: feature-group ablation. One registered experiment per (feature set, model).

    python experiments/feature_ablation/run_ablation.py [--skew] [--models ridge elasticnet xgboost]
                                                        [--feature-set-list base base+areas ...]
                                                        [--budget-minutes 25] [--dry-run]

Writes the registry only without ``--dry-run``. Configurations already in the registry
(same config_hash) are reused, not re-run. Output: a pivot (feature set x model -> cv_mean)
printed and saved to experiments/feature_ablation/ablation_<timestamp>.csv.

Finding (Agent A, 2026-11): the three anomaly flags are ALL ZERO in the development (train) data;
the anomalies (test Ids 2593, 2550, 1877) exist only in the test set. "base+anomaly_flags" therefore
cannot differ from "base", so it is excluded from the default list. It can still be requested
explicitly with --feature-set-list ... base+anomaly_flags.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from _ablation_common import LINEAR, MODELS, Runner, compare_text, model_spec

GROUPS = ["areas", "baths", "ages", "quality_interactions", "ordinal_quality"]
ALL_FS = "+".join(["base", *GROUPS])  # called "all" in the reports

HYPOTHESES = {
    "areas": "TotalSF and TotalPorchSF summarise scattered area columns; price is roughly proportional to total "
             "living area, so one clean total should help linear models most and trees little.",
    "baths": "A single weighted bath count (half baths = 0.5) condenses four sparse count columns into the "
             "quantity buyers care about; expected small gain for linear models.",
    "ages": "Age at sale (HouseAge, RemodAge, GarageAge) is more directly tied to depreciation than raw calendar "
            "years; expected gain for linear models, which cannot form YrSold - YearBuilt themselves.",
    "quality_interactions": "Quality times size (OverallQual x GrLivArea / TotalSF) captures that a quality "
                            "premium scales with house size; linear models cannot build this interaction.",
    "ordinal_quality": "Quality labels (Ex..Po) are ordered; numeric twins give one monotone coefficient instead "
                       "of five unrelated one-hot columns, adding strength to linear models.",
    "anomaly_flags": "Flags for implausible years (garage built after sale, sold before built, remodel before "
                     "build) could let a model separate data-entry errors. CAUTION: the flags are all zero in "
                     "the training data (anomalies exist only in the test set), so no effect is possible; "
                     "the run documents this rather than expecting a gain.",
}


def default_feature_sets() -> list[str]:
    sets = ["base"]
    sets += [f"base+{g}" for g in GROUPS]
    sets.append(ALL_FS)
    for g in GROUPS:
        sets.append("+".join(["base", *[x for x in GROUPS if x != g]]))
    return sets


def describe_set(fs: str) -> tuple[str, str]:
    """(kind, group): kind in base | single | all | loo | other."""
    extra = [g for g in fs.split("+") if g != "base"]
    if fs == "base":
        return "base", ""
    if fs == ALL_FS:
        return "all", ""
    if len(extra) == 1:
        return "single", extra[0]
    missing = [g for g in GROUPS if g not in extra]
    if len(missing) == 1 and set(extra) <= set(GROUPS):
        return "loo", missing[0]
    return "other", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set-list", nargs="+", default=default_feature_sets())
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--skew", action="store_true", help="add a log1p-skew pass for linear models on base and all")
    ap.add_argument("--budget-minutes", type=float, default=25.0)
    ap.add_argument("--dry-run", action="store_true", help="plumbing check on 150 rows; no CV, no registry write")
    args = ap.parse_args()

    runner = Runner(args.dry_run, args.budget_minutes)
    rows: dict[tuple[str, str], float] = {}

    def run(fs: str, model: str, skew: bool) -> None:
        kind, grp = describe_set(fs)
        builder, kwargs = model_spec(model, {"skew_log1p": True} if skew else None)
        key = (fs, model, skew)
        base_key = ("base", model, False)
        all_key = (ALL_FS, model, False)
        mult = "log1p on skewed numeric columns (in-fold)" if skew else "default preprocessing"
        title = f"Ablation: {fs}{' + skew_log1p' if skew else ''} / {model}"

        def interpret(res):
            ref = runner.scores.get(base_key)
            folds = res.fold_scores
            if key == base_key:
                return (f"Reference run for {model}: RMSE {res.mean:.4f} (std {res.std:.4f}). All deltas for "
                        f"this model are measured against it.", "retain")
            if ref is None:
                return (f"RMSE {res.mean:.4f}; base reference unavailable at this point.", "investigate")
            if skew:
                text, verdict = compare_text("skew_log1p", folds, ref["folds"], f"{model} on base without skew")
                if fs == ALL_FS and all_key in runner.scores:
                    t2, _ = compare_text("vs all without skew", folds, runner.scores[all_key]["folds"], "all")
                    text += " " + t2
                return text, {"better": "retain", "worse": "reject", "tie": "investigate"}[verdict]
            text, verdict = compare_text(fs, folds, ref["folds"], f"{model} on base")
            if kind == "loo" and all_key in runner.scores:
                t2, v2 = compare_text(f"removing {grp} from all", folds, runner.scores[all_key]["folds"], "all")
                text += " " + t2
                # removing a group that makes things worse => the group helps => retain it.
                return text, {"worse": "retain", "better": "reject", "tie": "investigate"}[v2]
            if fs == "base+anomaly_flags":
                return text + " The flags are constant (all zero) in the training data, so any difference is noise.", "reject"
            return text, {"better": "retain", "worse": "reject", "tie": "investigate"}[verdict]

        if kind == "base":
            hyp = "Reference configuration."
        elif kind == "all":
            hyp = "Engineered groups jointly add information that raw columns hide."
        elif kind == "single":
            hyp = HYPOTHESES[grp]
        elif kind == "loo":
            hyp = f"Leave-one-out of {grp}: removing it from 'all' should hurt if the group carries signal. " + HYPOTHESES[grp]
        else:
            hyp = "See title."
        if skew:
            hyp = "Skewed numeric columns (areas, porch sizes) bend the linear fit; log1p on those columns should help penalised linear models."
        rat = ("Skewness is learned in-fold; applies only to linear models." if skew else
               "Ablation isolates each engineered group's contribution under identical folds and models.")
        res = runner.get(
            key, fs, builder, kwargs, title=title, hypothesis=hyp, rationale=rat,
            implementation=f"features.FEATURE_SETS groups {fs.split('+')}; preprocessing: {mult}; "
                           f"model {builder} {kwargs}",
            interpret=interpret,
            tags=["feature_ablation", kind, *([grp] if grp else []), *(["skew"] if skew else [])])
        if res is not None:
            rows[(fs + ("|skew_log1p" if skew else ""), model)] = res["mean"]

    t0 = time.time()
    for fs in args.feature_set_list:
        print(f"== {fs}")
        for model in args.models:
            run(fs, model, False)
    if args.skew:
        print("== skew pass (linear models only)")
        for fs in ("base", ALL_FS):
            for model in [m for m in args.models if m in LINEAR]:
                run(fs, model, True)

    pivot = pd.Series(rows).unstack().reindex(columns=args.models)
    pivot.index.name = "feature_set"
    print("\nPivot: feature_set x model -> cv_mean (RMSE on log1p, lower is better)")
    print(pivot.round(5).to_string())
    if runner.skipped_for_budget:
        print(f"\nNOT RUN (budget): {runner.skipped_for_budget}")
    if args.dry_run:
        print("\n[dry-run] nothing saved, registry untouched.")
    else:
        out = Path(__file__).parent / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        pivot.to_csv(out)
        print(f"saved {out}  (elapsed {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
