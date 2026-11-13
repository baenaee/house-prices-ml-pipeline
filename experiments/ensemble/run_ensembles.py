"""C7: blends, nested stacks and a naive average of the best registered base models.

Real run (writes the registry; only when the director says GO):

    python experiments/ensemble/run_ensembles.py --select-best --feature-set base+areas+baths+ages+quality_interactions+ordinal_quality --train-row-filter keep_all --budget-minutes 40

Plumbing check (150 rows, never registers anything, never writes files):

    python experiments/ensemble/run_ensembles.py --dry-run --fast-trees

How the bases are chosen
    ``--select-best`` (default): per family (ridge, lasso, elasticnet, xgboost, lightgbm, catboost) the
    registered single-model experiment with the lowest ``cv_mean`` whose feature_set and train_row_filter
    match the arguments, single 5-fold repeat. Linear families must use ``skew_log1p`` and tree families
    must not (director ruling). Ensemble records are never candidates. ``--base-ids E0xx,E0yy`` overrides.

Experiments (all through ``_common.run_and_register``, agent "ensemble-researcher", ids assigned by the registry)
    naive equal-weight average of best linear + best tree (reference);
    blends with weights learned INSIDE fit (constrained least squares on inner OOF predictions):
    all bases, linear-only, tree-only, best-linear + best-tree;
    nested stacks (Ridge meta-learner, passthrough False): same four groups.
    Every record's builder is ``houseprices.ensemble:build_*_from_specs`` with JSON-only kwargs, so
    ``python -m houseprices.evaluate --experiment <id>`` rebuilds it. Configs already registered are skipped.

Interpretation rule: each ensemble is compared with the best single base among its own members (and with the
best base overall), on the SAME folds, using the mean of per-fold differences and its paired standard error
(std of differences / sqrt(k)). Better than one SE -> retain; worse than one SE -> reject; otherwise
investigate. The director re-tests finalists with RepeatedKFold 5x3.

F17: the "optimistic" diagnostics in houseprices.ensemble return ``OptimisticScore`` objects which
``run_and_register`` cannot accept; this script never calls them and never registers anything optimistic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import _common  # noqa: E402

from houseprices import ensemble, registry, validation  # noqa: E402

AGENT = "ensemble-researcher"
DRY_RUN_ROWS = 150
WORKING_SET = "base+areas+baths+ages+quality_interactions+ordinal_quality"
FAMILIES = {  # family -> (builder function name, kind)
    "ridge": ("build_ridge", "linear"),
    "lasso": ("build_lasso", "linear"),
    "elasticnet": ("build_elasticnet", "linear"),
    "xgboost": ("build_xgboost", "tree"),
    "lightgbm": ("build_lightgbm", "tree"),
    "catboost": ("build_catboost", "tree"),
}
TAIL_TO_FAMILY = {v[0]: k for k, v in FAMILIES.items()}
FAST_TREE_PARAMS = {"xgboost": {"n_estimators": 20}, "lightgbm": {"n_estimators": 20}, "catboost": {"iterations": 20}}
# Fit cost of one nested level relative to a single 5-fold CV of the same bases: per outer fold the
# inner 5-fold CV fits 5 models on 64% of the rows each, plus one refit on 80%, about 5.5x.
NESTED_COST_FACTOR = 5.5
DRY_DEFAULTS = {  # dry-run only, when the registry has no matching base yet
    "ridge": {"alpha": 10.0}, "lasso": {"alpha": 0.001}, "elasticnet": {"alpha": 0.001, "l1_ratio": 0.5},
    "xgboost": {}, "lightgbm": {}, "catboost": {},
}


# ---------------------------------------------------------------------------
# Base selection
# ---------------------------------------------------------------------------


class Base:
    def __init__(self, family: str, exp_id: str, builder: str, kwargs: dict, cv_mean: float,
                 folds: list[float], fit_seconds: float):
        self.family, self.id, self.builder, self.kwargs = family, exp_id, builder, kwargs
        self.cv_mean, self.folds, self.fit_seconds = cv_mean, folds, fit_seconds
        self.kind = FAMILIES[family][1]

    @property
    def spec(self) -> dict:
        return {"builder": self.builder, "builder_kwargs": self.kwargs}


def _family(builder: str) -> str | None:
    if not builder.startswith("houseprices.models:"):
        return None
    return TAIL_TO_FAMILY.get(builder.split(":")[-1])


def _skew(kwargs: dict) -> bool:
    return bool((kwargs.get("preprocessor_kwargs") or {}).get("skew_log1p", False))


def _from_record(rec: dict, family: str) -> Base:
    return Base(family, rec["id"], rec["builder"], rec["builder_kwargs"], float(rec["cv_mean"]),
                [float(x) for x in rec["fold_scores"]], float(rec.get("fit_seconds") or 0.0))


def select_best(feature_set: str, train_row_filter: str) -> list[Base]:
    reg = registry.load_registry()
    records = [] if reg.empty else reg.to_dict("records")
    out: list[Base] = []
    for fam, (_, kind) in FAMILIES.items():
        best = None
        for rec in records:
            if _family(rec["builder"]) != fam or rec.get("cv_mean") is None or pd.isna(rec.get("cv_mean")):
                continue
            if rec["feature_set"] != feature_set or rec["train_row_filter"] != train_row_filter:
                continue
            if rec["cv_n_repeats"] != 1 or len(rec["fold_scores"]) != rec["cv_n_splits"]:
                continue
            if (kind == "linear") != _skew(rec["builder_kwargs"]):  # linear needs skew_log1p, trees must not
                continue
            if best is None or rec["cv_mean"] < best["cv_mean"]:
                best = rec
        if best is not None:
            out.append(_from_record(best, fam))
        else:
            print(f"  no registered {fam} experiment for feature_set={feature_set!r}, filter={train_row_filter!r}")
    return out


def bases_from_ids(ids: list[str]) -> tuple[list[Base], str, str]:
    bases, sets = [], set()
    for i in ids:
        rec = registry.get_experiment(i)
        fam = _family(rec["builder"])
        if fam is None:
            raise ValueError(f"{i} is not a single-model experiment from houseprices.models")
        bases.append(_from_record(rec, fam))
        sets.add((rec["feature_set"], rec["train_row_filter"]))
    if len(sets) != 1:
        raise ValueError(f"--base-ids must share one feature_set/train_row_filter, got {sets}")
    fs, flt = next(iter(sets))
    return bases, fs, flt


def dry_run_fallback(bases: list[Base]) -> list[Base]:
    """Dry-run only: fill missing families with default-parameter bases so every code path is exercised."""
    have = {b.family for b in bases}
    for fam, (tail, kind) in FAMILIES.items():
        if fam not in have:
            kw = dict(DRY_DEFAULTS[fam])
            if kind == "linear":
                kw["preprocessor_kwargs"] = {"skew_log1p": True}
            bases.append(Base(fam, "dry-default", f"houseprices.models:{tail}", kw, float("nan"), [], 1.0))
            print(f"  [dry-run] {fam}: using default parameters (nothing registered yet)")
    return bases


# ---------------------------------------------------------------------------
# Statistics and interpretation
# ---------------------------------------------------------------------------


def paired(scores: list[float], ref: list[float]) -> tuple[float, float]:
    """(mean of per-fold differences scores-ref, paired SE). Negative = better than ref."""
    d = np.asarray(scores) - np.asarray(ref)
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se


def verdict(delta: float, se: float) -> str:
    if not np.isfinite(se) or abs(delta) <= se:
        return "investigate"
    return "retain" if delta < 0 else "reject"


def make_interpret(label: str, members: list[Base], all_bases: list[Base]):
    best_member = min(members, key=lambda b: b.cv_mean)
    best_all = min(all_bases, key=lambda b: b.cv_mean)

    def interpret(res: validation.CVResult) -> tuple[str, str]:
        d, se = paired(res.fold_scores, best_member.folds)
        text = (f"{label}: RMSE {res.mean:.4f} (std {res.std:.4f}). Best single member {best_member.family} "
                f"{best_member.id} scores {best_member.cv_mean:.4f}; paired difference {d:+.4f} (SE {se:.4f}) "
                f"on the same folds, {'beyond' if abs(d) > se else 'within'} one paired standard error.")
        if best_all.id != best_member.id:
            d2, se2 = paired(res.fold_scores, best_all.folds)
            text += (f" Against the best base overall ({best_all.family} {best_all.id}, {best_all.cv_mean:.4f}): "
                     f"{d2:+.4f} (SE {se2:.4f}).")
        dec = verdict(d, se)
        text += " Decision by the one-SE rule: " + {
            "retain": "lower RMSE than the best member by more than one SE, retain as a finalist.",
            "reject": "higher RMSE than the best member by more than one SE, reject.",
            "investigate": "indistinguishable from the best single member; the extra complexity is not "
                           "justified yet, investigate with RepeatedKFold 5x3.",
        }[dec]
        return text, dec

    return interpret


# ---------------------------------------------------------------------------
# Experiment plan
# ---------------------------------------------------------------------------


def plan(bases: list[Base]) -> list[dict]:
    lin = [b for b in bases if b.kind == "linear"]
    tre = [b for b in bases if b.kind == "tree"]
    groups: dict[str, list[Base]] = {}
    if lin and tre:
        groups["best_linear+best_tree"] = [min(lin, key=lambda b: b.cv_mean), min(tre, key=lambda b: b.cv_mean)]
    if len(bases) > 2:
        groups["all"] = list(bases)
    if len(lin) > 1:
        groups["linear_only"] = lin
    if len(tre) > 1:
        groups["tree_only"] = tre
    items: list[dict] = []
    if "best_linear+best_tree" in groups:
        items.append(dict(kind="average", group="best_linear+best_tree", members=groups["best_linear+best_tree"]))
    for kind in ("blend", "stack"):
        for g in ("best_linear+best_tree", "all", "linear_only", "tree_only"):
            if g in groups:
                items.append(dict(kind=kind, group=g, members=groups[g]))
    for it in items:
        specs = [b.spec for b in it["members"]]
        if it["kind"] == "blend":
            it["builder"] = "houseprices.ensemble:build_blend_from_specs"
            it["kwargs"] = {"specs": specs, "weights": None}
        elif it["kind"] == "stack":
            it["builder"] = "houseprices.ensemble:build_stack_from_specs"
            it["kwargs"] = {"specs": specs, "meta_alpha": 1.0, "passthrough": False}
        else:
            it["builder"] = "houseprices.ensemble:build_average_from_specs"
            it["kwargs"] = {"specs": specs}
        base_cost = sum(b.fit_seconds for b in it["members"])
        it["est_seconds"] = base_cost * (1.0 if it["kind"] == "average" else NESTED_COST_FACTOR)
        it["name"] = f"{it['kind']}[{it['group']}]"
    return items


def describe(it: dict) -> tuple[str, str, str]:
    mem = ", ".join(f"{b.family}({b.id})" for b in it["members"])
    if it["kind"] == "average":
        hyp = ("Equal-weight average of the best linear and best tree model. Their errors differ in kind, so the "
               "average should beat both if the errors are not too correlated. Naive reference for the learned blends.")
        impl = f"BlendingRegressor with fixed weights 1/k over {mem}."
    elif it["kind"] == "blend":
        hyp = (f"A blend of {it['group']} bases with weights learned from inner out-of-fold predictions should "
               "beat the best single member when member errors are not perfectly correlated.")
        impl = ("BlendingRegressor(weights=None): inner 5-fold OOF inside fit, non-negative weights summing to one, "
                f"bases refit on all received rows. Members: {mem}.")
    else:
        hyp = (f"A nested stack of {it['group']} bases with a Ridge meta-learner can reweight members "
               "more flexibly than a convex blend, at the cost of a second learned layer.")
        impl = ("NestedStackingRegressor: inner 5-fold OOF inside fit, Ridge(alpha=1) meta-learner, passthrough False, "
                f"bases refit on all received rows. Members: {mem}.")
    rat = ("Meta-learner or weights are learned only from out-of-fold predictions inside each outer-train fold, so "
           "the outer validation fold never influences them (see houseprices.ensemble).")
    return hyp, rat, impl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ids", default=None, help="comma-separated registry ids, overrides --select-best")
    ap.add_argument("--select-best", action="store_true", default=True, help="(default) pick best per family")
    ap.add_argument("--feature-set", default=WORKING_SET)
    ap.add_argument("--train-row-filter", default="keep_all")
    ap.add_argument("--budget-minutes", type=float, default=40.0)
    ap.add_argument("--dry-run", action="store_true", help=f"{DRY_RUN_ROWS} rows; no registry write, no files saved")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="run only the diversity analysis (writes the diversity CSV, registers nothing)")
    ap.add_argument("--fast-trees", action="store_true", help="dry-run only: tiny n_estimators/iterations for trees")
    args = ap.parse_args()
    if args.fast_trees and not args.dry_run:
        ap.error("--fast-trees changes model configs and is only allowed with --dry-run")

    feature_set, row_filter = args.feature_set, args.train_row_filter
    if args.base_ids:
        bases, feature_set, row_filter = bases_from_ids([s.strip() for s in args.base_ids.split(",") if s.strip()])
    else:
        print(f"Selecting best registered base per family (feature_set={feature_set}, filter={row_filter})")
        bases = select_best(feature_set, row_filter)
    if args.dry_run:
        _common.set_dry_run(DRY_RUN_ROWS)
        bases = dry_run_fallback(bases)
        if args.fast_trees:
            for b in bases:
                if b.family in FAST_TREE_PARAMS:
                    b.kwargs = {**b.kwargs, **FAST_TREE_PARAMS[b.family]}
    if len(bases) < 2 or {b.kind for b in bases} != {"linear", "tree"}:
        sys.exit(f"need at least one linear and one tree base, found {[b.family for b in bases]}; "
                 "wait for the linear grids and Optuna stage 1 to register their best configurations")

    print("\nChosen bases:")
    for b in bases:
        print(f"  {b.family:<11} {b.id:<11} cv_mean={b.cv_mean:.5f}  fit={b.fit_seconds:.0f}s  {b.kwargs}")
    items = plan(bases)

    diversity_cost = sum(b.fit_seconds for b in bases)
    total = diversity_cost + sum(it["est_seconds"] for it in items)
    print(f"\nTime estimate (registered fit_seconds; nested ensembles ~{NESTED_COST_FACTOR}x their bases' CV time):")
    print(f"  diversity OOF pass: {diversity_cost / 60:.1f} min")
    for it in items:
        print(f"  {it['name']:<34} {it['est_seconds'] / 60:5.1f} min")
    print(f"  TOTAL ~{total / 60:.1f} min (budget {args.budget_minutes:.0f} min)"
          + ("  -> exceeds the budget, later experiments will be skipped" if total / 60 > args.budget_minutes else ""))

    t_start = time.time()
    deadline = t_start + args.budget_minutes * 60

    # ---- diversity analysis ----------------------------------------------------------------
    # DIAGNOSTIC ONLY: these single-pass OOF predictions describe how the bases err. They are never used
    # to fit anything (weights, meta-learner, selection) that is then scored on the same rows; the
    # ensembles below learn their weights inside each outer-train fold from inner OOF predictions.
    print("\nDiversity analysis (single-pass OOF of each base, same folds as the registry)")
    dev = _common.get_data(feature_set)
    oof = {}
    for b in bases:
        res = _common.run_cv(b.builder, b.kwargs, feature_set, row_filter)
        oof[b.family] = res.oof_pred
        if args.dry_run:
            b.folds, b.cv_mean = list(res.fold_scores), res.mean
        else:
            print(f"  {b.family}: recomputed cv_mean {res.mean:.6f} vs registered {b.cv_mean:.6f} ({b.id})")
    oof_df = pd.DataFrame(oof)
    y = dev.y_log.to_numpy()
    pc = ensemble.prediction_correlation(oof_df)
    rc = ensemble.residual_correlation(oof_df, y)
    rep = ensemble.diversity_report(oof_df, y)
    print("\nRMSE and mean residual correlation with the other bases:\n" + rep.round(4).to_string())
    print("\nResidual correlation:\n" + rc.round(3).to_string())
    print("\nPrediction correlation:\n" + pc.round(3).to_string())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent
    if not args.dry_run:
        path = out_dir / f"diversity_{stamp}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("# diversity_report\n")
            rep.to_csv(fh)
            fh.write("\n# residual_correlation\n")
            rc.to_csv(fh)
            fh.write("\n# prediction_correlation\n")
            pc.to_csv(fh)
        print(f"saved {path}")

    # ---- ensembles ---------------------------------------------------------------------------
    if args.diagnostics_only:
        print("\n[diagnostics-only] no ensembles run, registry untouched.")
        return
    print("\nEnsembles")
    rows = []
    for it in items:
        hyp, rat, impl = describe(it)
        title = f"Ensemble: {it['kind']} of {it['group']} ({'+'.join(b.family for b in it['members'])})"
        interpret = make_interpret(it["name"], it["members"], bases)
        if not args.dry_run:
            existing = _common.find_registered(it["builder"], it["kwargs"], feature_set, row_filter)
            if existing:
                print(f"  skip {it['name']}: already registered as {existing}")
                continue
            if time.time() + it["est_seconds"] > deadline:
                print(f"  BUDGET: not run (would exceed {args.budget_minutes:.0f} min): {it['name']}")
                continue
        t0 = time.time()
        res = _common.run_cv(it["builder"], it["kwargs"], feature_set, row_filter)
        best_member = min(it["members"], key=lambda b: b.cv_mean)
        d, se = paired(res.fold_scores, best_member.folds)
        if args.dry_run:
            text, dec = interpret(res)
            print(f"  [dry-run] {it['name']}: cv_mean={res.mean:.4f} ({time.time() - t0:.0f}s) -> {dec}\n    {text}")
            exp_id = "dry"
        else:
            exp = _common.run_and_register(
                "", title, hyp, rat, impl, it["builder"], it["kwargs"], interpret, feature_set=feature_set,
                train_row_filter=row_filter, tags=["ensemble", it["kind"], it["group"]], result=res, agent=AGENT)
            exp_id, dec = exp.id, exp.decision
            print(f"  ran {it['name']}: {exp.id} cv_mean={res.mean:.4f} vs best member {best_member.cv_mean:.4f} "
                  f"(delta {d:+.4f}, SE {se:.4f}) in {time.time() - t0:.0f}s -> {dec}")
        rows.append(dict(id=exp_id, name=it["name"], cv_mean=res.mean, best_member=best_member.id,
                         best_member_cv=best_member.cv_mean, delta=d, paired_se=se, decision=dec))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        print("\nSummary (delta < 0 means better than the best member):\n" + summary.round(5).to_string(index=False))
    if args.dry_run:
        print("\n[dry-run] nothing registered, no files saved.")
    elif not summary.empty:
        path = out_dir / f"ensembles_{stamp}.csv"
        summary.to_csv(path, index=False)
        print(f"saved {path}  (elapsed {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
