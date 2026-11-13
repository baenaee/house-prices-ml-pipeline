"""C3: LotFrontage imputation comparison (all variants fitted inside each training fold).

    python experiments/feature_ablation/run_lotfrontage.py [--feature-set base] [--no-knn] [--dry-run]

Variants (``preprocessor_kwargs={"lotfrontage": ...}``): neighborhood_median (GroupMedianImputer),
global_median (plain median), knn (5 nearest training houses on LotArea, 1stFlrSF, GrLivArea,
OverallQual, YearBuilt). Models: Ridge alpha=10 and XGBoost defaults.
"""

from __future__ import annotations

import argparse

from _ablation_common import Runner, compare_text, model_spec

REF_VARIANT = "neighborhood_median"
VARIANTS = {
    "neighborhood_median": "Houses in the same neighborhood have similar frontage; the Neighborhood median is the "
                           "usual remedy and should beat a single global median.",
    "global_median": "Control: a single median ignores location; expected to be slightly worse or equal.",
    "knn": "Frontage relates to lot area (and size and age of the house); nearest neighbours on those attributes "
           "may recover frontage better than a Neighborhood median.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", default="base")
    ap.add_argument("--no-knn", action="store_true")
    ap.add_argument("--budget-minutes", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true", help="plumbing check on 150 rows; no CV, no registry write")
    args = ap.parse_args()

    variants = [v for v in VARIANTS if not (args.no_knn and v == "knn")]
    runner = Runner(args.dry_run, args.budget_minutes)
    for model in ("ridge", "xgboost"):
        for v in variants:
            builder, kwargs = model_spec(model, {"lotfrontage": v})
            key = (args.feature_set, model, v)
            ref_key = (args.feature_set, model, REF_VARIANT)

            def interpret(res, v=v, model=model, key=key, ref_key=ref_key):
                ref = runner.scores.get(ref_key)
                if key == ref_key or ref is None:
                    return f"Reference variant for {model}: RMSE {res.mean:.4f} (std {res.std:.4f}).", "retain"
                text, verdict = compare_text(v, res.fold_scores, ref["folds"], f"{REF_VARIANT} ({model})")
                if v == "knn":
                    decision = {"better": "retain", "worse": "reject", "tie": "investigate"}[verdict]
                else:
                    decision = {"better": "investigate", "worse": "reject", "tie": "reject"}[verdict]
                return text, decision

            runner.get(key, args.feature_set, builder, kwargs,
                       title=f"LotFrontage {v} / {model} / {args.feature_set}",
                       hypothesis=VARIANTS[v],
                       rationale="LotFrontage has about 17% missing values; the fill strategy is a modelling "
                                 "choice that must be fitted inside each training fold.",
                       implementation=f"preprocess.make_preprocessor(lotfrontage={v!r}); model {builder} {kwargs}",
                       interpret=interpret, tags=["lotfrontage", v])
    print("\nSummary (cv_mean, RMSE log1p):")
    for (_, model, v), r in runner.scores.items():
        print(f"  {model:8s} {v:20s} {r['mean']:.5f}  [{r['id']}]")
    if args.dry_run:
        print("[dry-run] nothing saved, registry untouched.")


if __name__ == "__main__":
    main()
