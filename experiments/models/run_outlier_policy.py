"""C2: compare three training-row (outlier) policies for three models, on one feature set.

    python experiments/models/run_outlier_policy.py --feature-set base

Policies are applied to the TRAINING part of each fold only; validation folds are always
complete, so scores are comparable across policies. 9 experiments are registered
(3 policies x Ridge / ElasticNet / XGBoost). Cells whose config_hash is already in the
registry are not re-run; their registered fold scores are reused for the comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _common  # noqa: E402
from houseprices import registry  # noqa: E402

POLICIES = ["keep_all", "drop_candidate_outliers", "drop_GrLivArea_gt_4000"]
MODELS = {
    "Ridge": ("houseprices.models:build_ridge", {"alpha": 10.0}, "alpha=10"),
    "ElasticNet": ("houseprices.models:build_elasticnet", {"alpha": 0.001, "l1_ratio": 0.5}, "alpha=0.001, l1_ratio=0.5"),
    "XGBoost": ("houseprices.models:build_xgboost", {}, "library defaults"),
}
Q4 = ("Auditor note Q4: drop_GrLivArea_gt_4000 also removes Ids 692 and 1183 (legitimate expensive houses) "
      "besides the two EDA candidates 524 and 1299, so it is a distinct policy, not a proxy for the Id list.")


def paired(scores: list[float], ref: list[float]) -> tuple[float, float]:
    """Mean and standard error of the per-fold differences (scores - ref), same folds."""
    d = np.asarray(scores) - np.asarray(ref)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="base")
    ap.add_argument("--dry-run-rows", type=int, default=None, help="plumbing check on first N rows; nothing is registered")
    args = ap.parse_args()
    fs = args.feature_set
    dry = args.dry_run_rows is not None
    if dry:
        _common.set_dry_run(args.dry_run_rows)

    table = {}      # (policy, model) -> (mean, std)
    for mname, (builder, kwargs, desc) in MODELS.items():
        # 1) obtain fold scores for every policy (reuse registered ones, run the rest)
        results, reused = {}, {}
        for pol in POLICIES:
            eid = None if dry else _common.find_registered(builder, kwargs, fs, pol)
            if eid:
                rec = registry.get_experiment(eid)
                reused[pol] = eid
                results[pol] = list(rec["fold_scores"])
                print(f"[skip] {mname}/{pol} already registered as {eid}; reusing its fold scores")
            else:
                r = _common.run_cv(builder, kwargs, fs, pol)
                results[pol] = r.fold_scores
                results[pol + "#r"] = r
        means = {p: float(np.mean(results[p])) for p in POLICIES}
        best = min(means, key=means.get)
        base = results["keep_all"]

        # 2) register the missing cells, with interpretations that quote the numbers
        for pol in POLICIES:
            m, s = means[pol], float(np.std(results[pol], ddof=1))
            table[(pol, mname)] = (m, s)
            if pol in reused or dry:
                continue
            d_keep, se_keep = paired(results[pol], base)
            d_best, se_best = paired(results[pol], results[best])
            if pol == "keep_all":
                cmp_txt = "Reference policy for this model; the other two are measured against it."
            else:
                sig = "exceeds" if abs(d_keep) > se_keep else "does not exceed"
                direction = "better (lower)" if d_keep < 0 else "worse (higher)" if d_keep > 0 else "identical"
                cmp_txt = (f"Versus keep_all: paired fold difference {d_keep:+.4f} (SE {se_keep:.4f}), {direction}; "
                           f"the difference {sig} one paired standard error.")
            if pol == best:
                decision = "retain"
                tail = f"Best of the three policies for {mname} (mean {m:.4f})."
            else:
                decision = "reject" if d_best > se_best else "investigate"
                tail = (f"Worse than the best policy ({best}, {means[best]:.4f}) by {d_best:+.4f} "
                        f"(paired SE {se_best:.4f}); " +
                        ("clearly worse, rejected." if decision == "reject" else "within noise of the best, so not rejected."))
            interp = f"RMSE {m:.4f} (fold std {s:.4f}). {cmp_txt} {tail}"
            title = f"Outlier policy {pol} | {mname}"
            _common.run_and_register(
                exp_id="", title=title,
                hypothesis=(f"Training-fold-only row policy '{pol}' changes the CV RMSE of {mname} ({desc}) "
                            "relative to keeping every row. " + Q4),
                rationale=("Two extreme-area, low-price houses (Ids 524, 1299) can pull linear fits; trees are expected to be "
                           "less sensitive. The GrLivArea>4000 rule is evaluated separately because it also drops "
                           "legitimate expensive houses (Ids 692, 1183). " + Q4),
                implementation=(f"{builder}({kwargs}); feature_set={fs}; train_row_filter={pol} applied after the split "
                                "to training rows only; validation fold untouched."),
                builder=builder, builder_kwargs=kwargs, feature_set=fs, train_row_filter=pol,
                interpret=lambda r, t=interp, d=decision: (t, d), result=results[pol + "#r"],
                tags=["outlier-policy", mname.lower()],
                notes=f"Paired differences use identical folds (KFold 5, seed 42). Means across policies: "
                      + ", ".join(f"{p}={v:.4f}" for p, v in means.items()),
            )

    mean_tab = pd.DataFrame({m: {p: table[(p, m)][0] for p in POLICIES} for m in MODELS}).round(5)
    std_tab = pd.DataFrame({m: {p: table[(p, m)][1] for p in POLICIES} for m in MODELS}).round(5)
    print("\nCV RMSE (log scale), policy x model:\n", mean_tab.to_string())
    print("\nFold std:\n", std_tab.to_string())
    if dry:
        print("(dry run: nothing saved or registered)")
        return
    out = Path(__file__).parent / f"outlier_policy_{fs}.csv"
    mean_tab.rename_axis("policy").to_csv(out)
    std_tab.rename_axis("policy").to_csv(out.with_name(out.stem + "_std.csv"))
    print("saved", out)


if __name__ == "__main__":
    main()
