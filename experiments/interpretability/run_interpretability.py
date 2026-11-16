"""E1: interpretability and error analysis of one registered experiment (read-only).

    python experiments/interpretability/run_interpretability.py --experiment E060
    python experiments/interpretability/run_interpretability.py --experiment E061 --shap-experiment E060
    python experiments/interpretability/run_interpretability.py --experiment E060 --dry-run

What it does
------------
1. Out-of-fold (OOF) predictions on the development rows with the selection folds
   (KFold 5, shuffle, seed 42, the experiment's own training-row filter), and residual tables
   by SalePrice decile, GrLivArea bin and Neighborhood, plus the 15 worst rows.
2. Permutation importance on the VALIDATION part of each fold. For each fold the pipeline is
   fit on the training part, then each RAW input column (every column of the development
   frame except Id) is shuffled in the validation part and the increase in RMSE on the log
   target is recorded (3 repeats, mean and std across the 5 folds).
   Why validation and not training rows: a model scores best on rows it has seen, so shuffling
   a column in the training rows mostly measures how much the model memorised that column;
   on held-out rows the increase is the loss of real predictive information.
   Caveat: correlated inputs share credit. Permuting GrLivArea alone leaves TotalSF,
   1stFlrSF and the interaction features intact, so each of them looks less important than the
   whole group is. Read the ranking as "marginal damage of removing one column", not as a
   partition of importance.
3. SHAP values of a tree single model (``--shap-experiment``): the pipeline is fit on all
   development rows, the fitted preprocessor produces the model matrix, and the booster is
   explained with ``shap.TreeExplainer``. SHAP values describe how this fitted model uses its
   columns (in-sample), not how well it generalises; use section 2 for that. One-hot levels
   are summed back to their source column in a second table.

Safety: never writes the registry, never loads or scores the holdout (the development frame
is built with ``data.load_development_data`` and only ``dev.X``/``dev.y_log`` are used), never
calls ``score_holdout_once``, ``train`` or ``predict``.
Dry run: 150 rows, tiny models, outputs in a temp folder, no holdout split is created.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

from houseprices import config, data, registry, validation  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TREE_BUILDERS = ("build_xgboost", "build_lightgbm", "build_catboost")
KEY_COLS = ["Neighborhood", "GrLivArea", "OverallQual", "YearBuilt", "SaleCondition", "MSSubClass"]
GRLIV_BINS = [0, 1000, 1500, 2000, 2500, 3000, 4000, 1e9]
GRLIV_LABELS = ["<1000", "1000-1499", "1500-1999", "2000-2499", "2500-2999", "3000-3999", ">=4000"]
DPI = 130


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_experiment(exp_id: str, dry: bool) -> tuple[dict, dict, object]:
    rec = registry.get_experiment(exp_id)        # read-only
    kwargs = rec.get("builder_kwargs") or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    if dry:                                       # tiny models: plumbing only
        kwargs = dict(kwargs)
        for key in ("n_estimators", "iterations"):
            if key in kwargs:
                kwargs[key] = 20
    builder = data.resolve_callable(rec["builder"])
    return rec, kwargs, builder


def load_dev(feature_set: str, dry: bool):
    # with_holdout_split=False in dry mode: nothing is created or read from the holdout file.
    dev = data.load_development_data(feature_set, with_holdout_split=not dry)
    X, y = dev.X, dev.y_log
    if dry:
        X, y = X.head(150), y.head(150)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  figure {path.name} ({kb:.0f} KB)" + ("  [over 300 KB]" if kb > 300 else ""))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------
# 1 + 2: OOF predictions and permutation importance (one pass over the folds)
# ---------------------------------------------------------------------------


def oof_and_permutation(estimator, X, y, row_filter, n_repeats: int, seed: int):
    cv = validation.make_cv(config.CV_N_SPLITS, 1, config.SEED)
    raw_cols = [c for c in X.columns if c != "Id"]
    rng = np.random.default_rng(seed)
    oof = np.full(len(X), np.nan)
    fold_scores: list[float] = []
    imp = np.zeros((config.CV_N_SPLITS, len(raw_cols)))
    for fold, tr, va in validation.iter_folds(X, y, cv, row_filter):
        t0 = time.perf_counter()
        model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        Xv, yv = X.iloc[va], y.iloc[va]
        pred = np.asarray(model.predict(Xv), dtype=float)
        oof[va] = pred
        base = rmse(yv, pred)
        fold_scores.append(base)
        for j, col in enumerate(raw_cols):
            vals = []
            for _ in range(n_repeats):
                Xp = Xv.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                vals.append(rmse(yv, model.predict(Xp)))
            imp[fold, j] = np.mean(vals) - base
        print(f"  fold {fold}: RMSE {base:.5f}, permutation done in {time.perf_counter() - t0:.0f}s")
    perm = pd.DataFrame({"column": raw_cols, "importance_mean": imp.mean(axis=0),
                         "importance_std_across_folds": imp.std(axis=0, ddof=1)})
    perm["n_folds_positive"] = (imp > 0).sum(axis=0)
    perm = perm.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return oof, fold_scores, perm


def residual_tables(X, y, oof, tab: Path, tag: str):
    actual = np.expm1(y.to_numpy())
    res = y.to_numpy() - oof                      # log scale, positive = under-prediction
    df = pd.DataFrame({"Id": X["Id"].to_numpy(), "actual": actual, "pred": np.expm1(oof),
                       "y_log": y.to_numpy(), "pred_log": oof, "residual": res})
    for c in KEY_COLS:
        if c in X.columns:
            df[c] = X[c].to_numpy()

    def summarise(g):
        return pd.Series({"n": len(g), "rmse": float(np.sqrt(np.mean(g["residual"] ** 2))),
                          "mean_signed_residual": float(g["residual"].mean())})

    df["price_decile"] = pd.qcut(df["actual"], 10, labels=[f"D{i}" for i in range(1, 11)])
    by_dec = df.groupby("price_decile", observed=True).apply(summarise, include_groups=False)
    by_dec["price_min"] = df.groupby("price_decile", observed=True)["actual"].min()
    by_dec["price_max"] = df.groupby("price_decile", observed=True)["actual"].max()
    by_dec.reset_index().to_csv(tab / f"{tag}_residuals_by_price_decile.csv", index=False)

    out = {"decile": by_dec}
    if "GrLivArea" in df.columns:
        df["grliv_bin"] = pd.cut(df["GrLivArea"], GRLIV_BINS, labels=GRLIV_LABELS, right=False)
        by_gr = df.groupby("grliv_bin", observed=True).apply(summarise, include_groups=False)
        by_gr.reset_index().to_csv(tab / f"{tag}_residuals_by_grlivarea.csv", index=False)
        out["grliv"] = by_gr
    if "Neighborhood" in df.columns:
        by_nb = df.groupby("Neighborhood").apply(summarise, include_groups=False).sort_values("rmse", ascending=False)
        by_nb.reset_index().to_csv(tab / f"{tag}_residuals_by_neighborhood_all.csv", index=False)
        ext = pd.concat([by_nb.head(5).assign(group="worst5"), by_nb.tail(5).assign(group="best5")])
        ext.reset_index().to_csv(tab / f"{tag}_residuals_by_neighborhood_top_bottom5.csv", index=False)
        out["nb"] = ext
    worst = df.reindex(df["residual"].abs().sort_values(ascending=False).index).head(15)
    worst = worst[["Id", "actual", "pred", "residual"] + [c for c in KEY_COLS if c in df.columns]]
    worst.to_csv(tab / f"{tag}_worst15_rows.csv", index=False)
    out["worst"] = worst
    df.drop(columns=["price_decile"]).to_csv(tab / f"{tag}_oof_predictions.csv", index=False)
    out["df"] = df
    return out


def residual_figures(df: pd.DataFrame, by_dec: pd.DataFrame, fig_dir: Path, tag: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["actual"], df["pred"], s=10, alpha=0.5)
    lo, hi = df["actual"].min() * 0.9, df["actual"].max() * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("actual SalePrice (log axis)"), ax.set_ylabel("out-of-fold prediction (log axis)")
    ax.set_title(f"{title}: predicted vs actual")
    save_fig(fig, fig_dir / f"{tag}_pred_vs_actual.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["pred_log"], df["residual"], s=10, alpha=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("predicted log1p(SalePrice)"), ax.set_ylabel("residual = actual - predicted (log)")
    ax.set_title(f"{title}: residual vs predicted")
    save_fig(fig, fig_dir / f"{tag}_residual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(by_dec.index.astype(str), by_dec["rmse"], color="#4c78a8", label="RMSE")
    ax.plot(by_dec.index.astype(str), by_dec["mean_signed_residual"], "o-", color="#e45756", label="mean signed residual")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("SalePrice decile (D1 cheapest)"), ax.set_ylabel("log-scale error")
    ax.set_title(f"{title}: error by price decile")
    ax.legend()
    save_fig(fig, fig_dir / f"{tag}_residual_by_decile.png")


def barh(series: pd.Series, err: pd.Series | None, xlabel: str, title: str, path: Path):
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(s.index.astype(str), s.values, xerr=None if err is None else err.iloc[::-1].values, color="#4c78a8")
    ax.set_xlabel(xlabel), ax.set_title(title)
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# 3: SHAP
# ---------------------------------------------------------------------------


def source_column(name: str, raw_cols: list[str]) -> str:
    """Map a model column like 'categorical__Neighborhood_NridgHt' back to its raw column."""
    base = name.split("__", 1)[-1]
    if base in raw_cols:
        return base
    cands = [c for c in raw_cols if base.startswith(c + "_")]
    return max(cands, key=len) if cands else base


def shap_section(exp_id: str, X, y, dry: bool, fig_dir: Path, tab: Path, tag: str):
    import shap

    rec, kwargs, builder = load_experiment(exp_id, dry)
    if rec["builder"].split(":")[-1] not in TREE_BUILDERS:
        print(f"[shap] {exp_id} is not a single tree model ({rec['builder']}); skipped")
        return None
    pipe = builder(**kwargs)
    t0 = time.perf_counter()
    row_filter = data.get_train_row_filter(rec.get("train_row_filter", "keep_all"))
    keep = np.asarray(row_filter(X, y), dtype=bool)
    pipe.fit(X[keep], y[keep])
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    M = prep.transform(X)
    if hasattr(model, "model_"):
        raise SystemExit("[shap] native-categorical CatBoost is not supported by this script (one-hot models only)")
    names = [str(n) for n in prep.get_feature_names_out()]
    Mdf = pd.DataFrame(np.asarray(M, dtype=float), columns=names)
    sv = shap.TreeExplainer(model).shap_values(Mdf)
    sv = np.asarray(sv)
    print(f"[shap] {exp_id}: fitted and explained {Mdf.shape[1]} columns in {time.perf_counter() - t0:.0f}s")

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    raw_cols = [c for c in X.columns if c != "Id"]
    out = pd.DataFrame({"model_column": mean_abs.index, "mean_abs_shap": mean_abs.values,
                        "source_column": [source_column(n, raw_cols) for n in mean_abs.index]})
    out.to_csv(tab / f"{tag}_shap_mean_abs_model_columns.csv", index=False)
    agg = out.groupby("source_column")["mean_abs_shap"].sum().sort_values(ascending=False)
    agg.rename("sum_of_mean_abs_shap").reset_index().to_csv(tab / f"{tag}_shap_mean_abs_by_raw_column.csv", index=False)
    barh(mean_abs.head(25), None, "mean |SHAP| (log-price units)", f"{exp_id}: SHAP, top 25 model columns",
         fig_dir / f"{tag}_shap_top25_bar.png")
    barh(agg.head(25), None, "sum of mean |SHAP| over one-hot levels", f"{exp_id}: SHAP aggregated to raw columns",
         fig_dir / f"{tag}_shap_top25_raw_bar.png")

    top20 = list(mean_abs.index[:20])
    idx = [names.index(n) for n in top20]
    plt.figure(figsize=(8, 7))
    shap.summary_plot(sv[:, idx], Mdf[top20], max_display=20, show=False)
    plt.title(f"{exp_id}: SHAP beeswarm, top 20")
    save_fig(plt.gcf(), fig_dir / f"{tag}_shap_beeswarm.png")
    for k, n in enumerate(top20[:2], start=1):
        plt.figure(figsize=(6.5, 5))
        shap.dependence_plot(names.index(n), sv, Mdf, show=False, interaction_index=None)
        save_fig(plt.gcf(), fig_dir / f"{tag}_shap_dependence_{k}.png")
    return mean_abs, agg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="registry id (single model or ensemble)")
    ap.add_argument("--shap-experiment", default=None, help="registry id of a tree single model (default: --experiment)")
    ap.add_argument("--repeats", type=int, default=3, help="permutation repeats per column and fold")
    ap.add_argument("--dry-run", action="store_true", help="150 rows, tiny models, temp output dir")
    args = ap.parse_args()

    if args.dry_run:
        out_root = Path(tempfile.mkdtemp(prefix="interp_dry_"))
        fig_dir, tab = out_root / "figures", out_root / "interpretability"
    else:
        fig_dir, tab = ROOT / "outputs" / "figures", ROOT / "outputs" / "interpretability"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    tag = args.experiment
    t_start = time.perf_counter()

    rec, kwargs, builder = load_experiment(args.experiment, args.dry_run)
    X, y = load_dev(rec.get("feature_set", "base"), args.dry_run)
    row_filter = data.get_train_row_filter(rec.get("train_row_filter", "keep_all"))
    print(f"{args.experiment}: {rec.get('title')} | rows={len(X)} | feature_set={rec.get('feature_set')}"
          f"{' | DRY RUN -> ' + str(fig_dir.parent) if args.dry_run else ''}")

    print("[1/3] out-of-fold predictions and permutation importance")
    est = builder(**kwargs)
    oof, fold_scores, perm = oof_and_permutation(est, X, y, row_filter, args.repeats, seed=config.SEED)
    cv_mean = float(np.mean(fold_scores))
    registered = [float(s) for s in (rec.get("fold_scores") or [])]
    if not args.dry_run:
        ok = len(registered) == len(fold_scores) and np.allclose(registered, fold_scores, atol=2e-4)
        print(f"  CV mean {cv_mean:.5f} (registered {rec.get('cv_mean')}); fold scores match registry: {ok}")
        if not ok:
            print("  WARNING: fold scores differ from the registered ones")
    perm.to_csv(tab / f"{tag}_permutation_importance.csv", index=False)
    top = perm.head(25).set_index("column")
    barh(top["importance_mean"], top["importance_std_across_folds"], "increase in RMSE (log) when the column is shuffled",
         f"{tag}: permutation importance on validation folds (top 25)", fig_dir / f"{tag}_permutation_top25.png")

    tabs = residual_tables(X, y, oof, tab, tag)
    residual_figures(tabs["df"], tabs["decile"], fig_dir, tag, tag)
    print("\nResidual RMSE by price decile:\n", tabs["decile"].round(4).to_string())
    if "grliv" in tabs:
        print("\nBy GrLivArea bin:\n", tabs["grliv"].round(4).to_string())
    if "nb" in tabs:
        print("\nNeighborhood worst/best 5:\n", tabs["nb"].round(4).to_string())
    print("\nWorst 15 rows:\n", tabs["worst"].round(4).to_string(index=False))
    print("\nTop 10 permutation importance:\n", perm.head(10).round(5).to_string(index=False))

    print("[3/3] SHAP")
    shap_id = args.shap_experiment or args.experiment
    shap_res = shap_section(shap_id, X, y, args.dry_run, fig_dir, tab, shap_id)
    if shap_res is not None:
        print("\nTop 10 SHAP model columns:\n", shap_res[0].head(10).round(5).to_string())
        print("\nTop 10 SHAP raw columns:\n", shap_res[1].head(10).round(5).to_string())
    print(f"\ndone in {(time.perf_counter() - t_start) / 60:.1f} min; tables in {tab}, figures in {fig_dir}")


if __name__ == "__main__":
    main()
