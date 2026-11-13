"""C4: grid search of the linear family (Ridge, Lasso, ElasticNet) x scaler (robust, standard).

    python experiments/models/run_linear_family.py --feature-set base --train-row-filter keep_all

Every grid point is cross-validated with the shared protocol but NOT registered; the full
grid goes to experiments/models/linear_grid_<feature_set>_<filter>.csv. Only the best
Ridge, best Lasso and best ElasticNet are registered, each with the grid summary in notes.
ConvergenceWarnings are counted, not shown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).parent))
import _common  # noqa: E402
from houseprices import validation  # noqa: E402

SCALERS = ["robust", "standard"]
SKEW = {"skew_log1p": True}   # director ruling: every linear grid point uses the in-fold log1p of skewed numerics
FAMILIES = {
    "Ridge": ("houseprices.models:build_ridge", [{"alpha": float(a)} for a in np.logspace(-1, 2.5, 15)]),
    "Lasso": ("houseprices.models:build_lasso", [{"alpha": float(a)} for a in np.logspace(-4.5, -1.5, 13)]),
    "ElasticNet": ("houseprices.models:build_elasticnet",
                   [{"alpha": float(a), "l1_ratio": l1} for a in np.logspace(-4, -1.5, 8)
                    for l1 in (0.1, 0.3, 0.5, 0.7, 0.9)]),
}


def run_grid(feature_set: str, row_filter: str, thin: int = 1) -> pd.DataFrame:
    rows = []
    for fam, (builder, grid) in FAMILIES.items():
        for scaler in SCALERS:
            for params in grid[::thin]:
                kwargs = {**params, "scaler": scaler, "preprocessor_kwargs": SKEW}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    r = _common.run_cv(builder, kwargs, feature_set, row_filter)
                n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                rows.append(dict(
                    model=fam, builder=builder, params=json.dumps(params), scaler=scaler,
                    alpha=params["alpha"], l1_ratio=params.get("l1_ratio", np.nan),
                    cv_mean=r.mean, cv_std=r.std,
                    fold_scores=json.dumps([round(s, 6) for s in r.fold_scores]),
                    seconds=round(r.fit_seconds, 2), convergence_warnings=n_conv))
        print(f"  {fam}: done ({sum(x['model'] == fam for x in rows)} points)", flush=True)
    return pd.DataFrame(rows)


def curve_text(sub: pd.DataFrame, se: float) -> str:
    """Describe the alpha curve of one scaler (for ElasticNet: at the winning l1_ratio)."""
    c = sub.sort_values("alpha").reset_index(drop=True)
    i = int(c["cv_mean"].idxmin())
    n = len(c)
    pos = ("at the LOW edge of the grid" if i == 0 else
           "at the HIGH edge of the grid" if i == n - 1 else f"in the interior (point {i + 1} of {n})")
    flat = int((c["cv_mean"] <= c["cv_mean"].min() + se).sum())
    shape = "flat" if flat >= max(3, n // 3) else "sharply peaked"
    return (f"minimum {c['cv_mean'].min():.4f} at alpha={c.loc[i, 'alpha']:.4g}, {pos}; "
            f"lowest-alpha end {c['cv_mean'].iloc[0]:.4f}, highest-alpha end {c['cv_mean'].iloc[-1]:.4f}; "
            f"{flat} of {n} points lie within one standard error ({se:.4f}) of the minimum, "
            f"so the curve is {shape} near the optimum")


def as_result(best: pd.Series, folds: np.ndarray) -> validation.CVResult:
    """Rebuild a CVResult from the stored grid row so the best point is not fit twice."""
    return validation.CVResult(list(map(float, folds)), np.array([]), float(best["seconds"]))


def register_best(fam: str, builder: str, g: pd.DataFrame, fs: str, flt: str, out_name: str) -> None:
    best = g.loc[g["cv_mean"].idxmin()]
    params, scaler = json.loads(best["params"]), best["scaler"]
    kwargs = {**params, "scaler": scaler, "preprocessor_kwargs": SKEW}
    eid = _common.find_registered(builder, kwargs, fs, flt)
    if eid:
        print(f"[skip] best {fam} {kwargs} already registered as {eid}")
        return
    folds = np.array(json.loads(best["fold_scores"]))
    per = {}
    for s in SCALERS:
        gs = g[g["scaler"] == s]
        per[s] = gs.loc[gs["cv_mean"].idxmin()]
    d = (np.array(json.loads(per["standard"]["fold_scores"])) -
         np.array(json.loads(per["robust"]["fold_scores"])))
    d_se = d.std(ddof=1) / np.sqrt(len(d))
    se = best["cv_std"] / np.sqrt(len(folds))

    sub = g[g["scaler"] == scaler]
    l1_txt = ""
    if fam == "ElasticNet":
        sub = sub[np.isclose(sub["l1_ratio"], params["l1_ratio"])]
        by_l1 = g.groupby("l1_ratio")["cv_mean"].min().round(4).to_dict()
        l1_txt = f" Best RMSE by l1_ratio: {by_l1}; the winner is l1_ratio={params['l1_ratio']}."
    curve = curve_text(sub, se)

    keys = ["alpha"] + (["l1_ratio"] if fam == "ElasticNet" else [])
    both = g.pivot_table(index=keys, columns="scaler", values="cv_mean")
    std_better = int((both["standard"] < both["robust"]).sum())
    a_r = json.loads(per["robust"]["params"])["alpha"]
    a_s = json.loads(per["standard"]["params"])["alpha"]
    verdict = "exceeds" if abs(d.mean()) > d_se else "does not exceed"
    scaler_txt = (f"Scaler comparison: best robust {per['robust']['cv_mean']:.4f} (alpha={a_r:.4g}) vs best "
                  f"standard {per['standard']['cv_mean']:.4f} (alpha={a_s:.4g}); paired fold difference "
                  f"standard minus robust {d.mean():+.4f} (SE {d_se:.4f}), which {verdict} one paired SE; "
                  f"standard is better at {std_better} of {len(both)} grid points.")
    summary = "; ".join(f"{s}: {per[s]['cv_mean']:.4f} @ {per[s]['params']}" for s in SCALERS)
    conv = int(g["convergence_warnings"].sum())
    notes = (f"Grid: {len(g)} CVs ({fam}, both scalers), row filter={flt}, feature set={fs}. "
             f"Best per scaler: {summary}. ConvergenceWarnings in this family's grid: {conv} "
             f"(max_iter=20000). Full grid: {out_name}.")
    interp = (f"Best {fam}: {kwargs}, RMSE {best['cv_mean']:.4f} (fold std {best['cv_std']:.4f}). "
              f"Alpha curve ({scaler} scaler): {curve}.{l1_txt} {scaler_txt}")
    extra = " x l1_ratio" if fam == "ElasticNet" else ""
    _common.run_and_register(
        exp_id="", title=f"{fam} best of grid ({scaler} scaler)",
        hypothesis=(f"A tuned {fam} penalty beats the untuned alpha of the baselines; the scaler choice may "
                    "matter because penalties act on scaled columns."),
        rationale=("Linear models on ~250 one-hot columns are very sensitive to penalty strength; a "
                   "log-spaced grid on the shared folds finds it cheaply."),
        implementation=f"{builder}({kwargs}); grid of {len(g)} points over alpha{extra} x scaler {SCALERS}.",
        builder=builder, builder_kwargs=kwargs, feature_set=fs, train_row_filter=flt,
        interpret=lambda r, t=interp: (t, "retain"),
        tags=["linear-family", "grid", fam.lower()], notes=notes, result=as_result(best, folds),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="base")
    ap.add_argument("--train-row-filter", default="keep_all")
    ap.add_argument("--dry-run-rows", type=int, default=None,
                    help="plumbing check: first N rows, every 4th grid point, no csv, no registration")
    args = ap.parse_args()
    fs, flt = args.feature_set, args.train_row_filter
    dry = args.dry_run_rows is not None
    if dry:
        _common.set_dry_run(args.dry_run_rows)

    t0 = time.perf_counter()
    grid = run_grid(fs, flt, thin=4 if dry else 1)
    total = time.perf_counter() - t0
    out = Path(__file__).parent / f"linear_grid_{fs}_{flt}.csv"
    if not dry:
        grid.to_csv(out, index=False)
        print("saved", out)
    print(f"grid: {len(grid)} CVs in {total / 60:.1f} min; "
          f"ConvergenceWarnings: {int(grid['convergence_warnings'].sum())}")
    best_rows = grid.loc[grid.groupby("model")["cv_mean"].idxmin()].set_index("model")
    print(best_rows[["scaler", "params", "cv_mean", "cv_std"]].to_string())
    if dry:
        return
    for fam, (builder, _) in FAMILIES.items():
        register_best(fam, builder, grid[grid["model"] == fam], fs, flt, out.name)


if __name__ == "__main__":
    main()
