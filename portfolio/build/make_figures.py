"""Build the portfolio figures (charts from the experiment registry + SVG flow diagrams).

Run from the repository root:
    python portfolio/build/make_figures.py

Every number is read from files already in the repository (registry, finalist table,
outlier-policy table, tuning trials, interpretability tables); nothing is re-computed
by a model. Outputs go to portfolio/figures/ as PNG (for PDFs and slides) and SVG.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape as xesc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "portfolio" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (validated reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def registry() -> dict[str, dict]:
    rows = {}
    with open(ROOT / "experiments" / "registry.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def title(ax, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


# ---------------------------------------------------------------- 1. results ladder
def fig_ladder(reg):
    steps = [
        ("E002", "OLS, numeric columns"),
        ("E003", "Ridge, one-hot categories"),
        ("E041", "Ridge + engineered features + skew fix"),
        ("E056", "Lasso, tuned alpha"),
        ("E060", "CatBoost, 25 Optuna trials"),
        ("E066", "CatBoost, 75 trials  (FINAL)"),
        ("E068", "Average Lasso + CatBoost"),
    ]
    labels = [f"{lab}  [{i}]" for i, lab in steps]
    vals = [reg[i]["cv_mean"] for i, _ in steps]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    colors = [ORANGE if i == "E066" else BLUE for i, _ in steps]
    y = range(len(steps))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 0.002, yy, f"{v:.4f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.set_xlim(0, 0.18)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("5-fold CV RMSE on log1p(SalePrice)  (lower is better)")
    title(ax, "From baseline to final model",
          f"Selection folds, seed 42. Predicting the mean scores {reg['E001']['cv_mean']:.3f} (E001), off the scale.")
    save(fig, "results_ladder")


# ---------------------------------------------------------------- 2. finalists
def fig_finalists():
    df = pd.read_csv(sorted((ROOT / "experiments" / "finalists").glob("finalists_*.csv"))[-1])
    names = {
        "E068": "Avg Lasso + CatBoost",
        "E061": "Avg (Stage 1 bases)",
        "E072": "Nested stack L + C",
        "E066": "CatBoost 75 trials (FINAL)",
        "E069": "NNLS blend L + C",
        "E060": "CatBoost 25 trials",
        "E067": "LightGBM 75 trials",
        "E056": "Lasso",
    }
    df = df.sort_values("fresh_cv_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, r in df.iterrows():
        ax.plot([r.selection_cv_mean, r.fresh_cv_mean], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(df.selection_cv_mean, df.index, s=55, color=BLUE, zorder=2, label="selection folds (seed 42, used to choose)")
    ax.scatter(df.fresh_cv_mean, df.index, s=55, color=ORANGE, zorder=2, label="fresh folds (5x3, seed 7, never used to choose)")
    for i, r in df.iterrows():
        ax.text(r.fresh_cv_mean + 0.0004, i, f"{r.fresh_cv_mean:.4f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(df.index, [names[s] for s in df.source_id], fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CV RMSE on log1p(SalePrice)")
    ax.set_xlim(0.115, 0.131)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False, fontsize=8.5)
    title(ax, "Every finalist got worse on folds it was not chosen on",
          f"Optimism (fresh minus selection) ranged {df.optimism.min():.4f} to {df.optimism.max():.4f}, mean {df.optimism.mean():.4f}.")
    save(fig, "finalists_fresh_folds")


# ---------------------------------------------------------------- 3. outlier policy
def fig_outliers():
    df = pd.read_csv(ROOT / "experiments" / "models" /
                     "outlier_policy_base+areas+baths+ages+quality_interactions+ordinal_quality.csv")
    df = df.set_index("policy")
    models = ["Ridge", "ElasticNet", "XGBoost"]
    pol = [("keep_all", "keep every row", BLUE),
           ("drop_candidate_outliers", "drop the 2 famous outliers (training folds only)", ORANGE),
           ("drop_GrLivArea_gt_4000", "drop all 4 houses > 4,000 sq ft (training folds only)", AQUA)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.26
    for k, (p, lab, c) in enumerate(pol):
        xs = [m + (k - 1) * (w + 0.02) for m in range(len(models))]
        vals = [df.loc[p, m] for m in models]
        ax.bar(xs, vals, width=w, color=c, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.0006, f"{v:.4f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(models)), models, color=INK)
    ax.set_ylim(0.12, 0.145)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("5-fold CV RMSE (validation folds keep every house)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    title(ax, "Removing outliers honestly made every model worse",
          "Y axis starts at 0.12 to show differences of a few thousandths. Registry E041/E042/E023 vs E049-E054.")
    save(fig, "outlier_policy")


# ---------------------------------------------------------------- 4. error by decile
def fig_deciles():
    df = pd.read_csv(ROOT / "outputs" / "interpretability" / "E066_residuals_by_price_decile.csv")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cols = [ORANGE if d in ("D1", "D10") else BLUE for d in df.price_decile]
    ax.bar(df.price_decile, df.rmse, color=cols, width=0.65)
    for x, v in zip(df.price_decile, df.rmse):
        ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("RMSE (out-of-fold)")
    ax.set_xlabel("Price decile (D1 = cheapest 10 %, D10 = most expensive 10 %)")
    ax.grid(axis="x", visible=False)
    title(ax, "Errors concentrate at the price extremes",
          "Final model E066, out-of-fold predictions. Cheapest decile over-predicted (-0.097), most expensive under-predicted (+0.072).")
    save(fig, "error_by_price_decile")


# ---------------------------------------------------------------- 5. optuna history
def fig_optuna():
    p = ROOT / "experiments" / "tuning" / \
        "stage1_catboost_base+areas+baths+ages+quality_interactions+ordinal_quality_trials.csv"
    df = pd.read_csv(p)
    df = df[df.state == "COMPLETE"].sort_values("number")
    best = df.value.cummin()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.scatter(df.number, df.value, s=18, color=MUTED, label="one trial (mean 5-fold CV)")
    ax.step(df.number, best, where="post", color=BLUE, lw=2, label="best so far")
    ax.axvline(24.5, color=INK2, lw=1, ls="--")
    ax.text(25.5, df.value.max() * 0.999, "Stage 2 starts\n(+50 trials)", fontsize=8, color=INK2, va="top")
    ax.set_ylim(0.117, min(0.135, df.value.max() + 0.001))
    ax.set_xlabel("Optuna trial number (TPE sampler)")
    ax.set_ylabel("CV RMSE")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    n_close = int((df.value <= df.value.min() + 0.003).sum())
    title(ax, "CatBoost tuning reached a plateau",
          f"{n_close} of {len(df)} trials are within 0.003 of the best ({df.value.min():.4f}).")
    save(fig, "catboost_optuna_history")


# ================================================================ SVG diagrams
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def box(x, y, w, h, title_txt, lines=(), fill="#f3f7fd", stroke=BLUE, tcol=INK, rx=8, fs=13):
    s = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"]
    ty = y + 22 if lines else y + h / 2 + 5
    s.append(f"<text x='{x + w / 2}' y='{ty}' text-anchor='middle' {FONT} font-size='{fs}' font-weight='700' fill='{tcol}'>{xesc(title_txt)}</text>")
    for k, ln in enumerate(lines):
        s.append(f"<text x='{x + w / 2}' y='{y + 40 + 15 * k}' text-anchor='middle' {FONT} font-size='11' fill='{INK2}'>{xesc(ln)}</text>")
    return "\n".join(s)


def arrow(x1, y1, x2, y2, col=INK2, dash=False):
    d = " stroke-dasharray='5,4'" if dash else ""
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='1.6' marker-end='url(#ah)'{d}/>"


def label(x, y, txt, size=11, col=INK2, anchor="middle", weight="400"):
    return f"<text x='{x}' y='{y}' text-anchor='{anchor}' {FONT} font-size='{size}' font-weight='{weight}' fill='{col}'>{xesc(txt)}</text>"


def svg_wrap(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}'>"
            f"<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
            f"<path d='M0,0 L10,5 L0,10 z' fill='{INK2}'/></marker></defs>"
            f"<rect width='{w}' height='{h}' fill='white'/>{body}</svg>")


def diagram_pipeline():
    """End-to-end ML flow of the project."""
    W, H = 1100, 640
    b = []
    G, GS = "#eef8f3", AQUA          # data stages
    Bf, Bs = "#f3f7fd", BLUE         # modelling stages
    O, OS = "#fdf1ec", ORANGE        # decision / guarded stages
    # row 1: data
    b.append(label(30, 30, "1  DATA (runs once, row-wise only)", 13, INK, "start", "700"))
    b.append(box(30, 45, 180, 70, "train.csv / test.csv", ["1,460 + 1,459 houses", "79 columns"], G, GS))
    b.append(box(250, 45, 190, 70, "Cleaning", ["17 deterministic rules", "semantic fills, no stats"], G, GS))
    b.append(box(480, 45, 190, 70, "Holdout split", ["219 rows locked (15 %)", "stratified by price decile"], O, OS))
    b.append(box(710, 45, 190, 70, "Feature groups", ["areas, baths, ages,", "quality x size interactions"], G, GS))
    b.append(box(940, 45, 140, 70, "Target", ["y = log1p(price)"], G, GS))
    for x1, x2 in ((210, 250), (440, 480), (670, 710), (900, 940)):
        b.append(arrow(x1, 80, x2, 80))
    # row 2: CV loop
    b.append(label(30, 165, "2  EXPERIMENTS on 1,241 development rows (everything that learns is fitted inside each training fold)", 13, INK, "start", "700"))
    b.append(f"<rect x='30' y='178' width='1050' height='190' rx='12' fill='none' stroke='{BLUE}' stroke-dasharray='6,5' stroke-width='1.5'/>")
    b.append(label(45, 200, "for each of 5 folds:  clone(pipeline) → fit on 4/5 → predict 1/5", 11.5, BLUE, "start", "700"))
    b.append(box(50, 215, 210, 135, "Preprocess (in fold)", ["LotFrontage: nbhd median", "log1p of skewed columns", "one-hot encoding", "scaling (linear models)"], Bf, BLUE))
    b.append(box(300, 215, 230, 135, "Model", ["OLS / Ridge / Lasso / ElasticNet", "XGBoost / LightGBM / CatBoost", "Optuna TPE: 180 trials", "ensembles (nested stacking)"], Bf, BLUE))
    b.append(box(570, 215, 200, 135, "Score", ["RMSE on log scale", "5 fold scores", "mean + paired diffs", "finite-check guard"], Bf, BLUE))
    b.append(box(810, 215, 250, 135, "Experiment registry", ["83 runs, E001-E083", "builder + args + seed + hash", "reproduce: evaluate --experiment", "217 tests, 36 audit findings"], Bf, BLUE))
    for x1, x2 in ((260, 300), (530, 570), (770, 810)):
        b.append(arrow(x1, 282, x2, 282))
    b.append(arrow(985, 115, 985, 176))
    # row 3: selection
    b.append(label(30, 415, "3  SELECTION, FREEZE AND DELIVERY", 13, INK, "start", "700"))
    b.append(box(30, 430, 200, 85, "8 finalists", ["re-scored on 15 fresh folds", "(seed 7): optimism 0.003"], O, OS))
    b.append(box(270, 430, 200, 85, "Pre-registered rule", ["ensemble must win by", ">= 0.001 and > 1 SE"], O, OS))
    b.append(box(510, 430, 180, 85, "Freeze E066", ["single CatBoost", "Git commit + audit OK"], O, OS))
    b.append(box(730, 430, 160, 85, "Holdout, once", ["the 219 locked rows", "RMSE 0.1065 (CV ~0.12)"], O, OS))
    b.append(box(930, 430, 150, 85, "Final fit", ["all 1,460 rows", "→ submission.csv"], O, OS))
    for x1, x2 in ((230, 270), (470, 510), (690, 730), (890, 930)):
        b.append(arrow(x1, 472, x2, 472))
    b.append(arrow(935, 368, 130, 428, dash=True))
    b.append(label(700, 300 + 105, "", 1))
    # legend
    for k, (f, s, t) in enumerate(((G, GS, "deterministic data step"), (Bf, BLUE, "fitted inside each CV fold"), (O, OS, "decision / guarded step"))):
        x = 30 + k * 260
        b.append(f"<rect x='{x}' y='560' width='22' height='16' rx='4' fill='{f}' stroke='{s}'/>")
        b.append(label(x + 30, 573, t, 11.5, INK2, "start"))
    b.append(label(810, 573, "holdout rows stay locked until step 3", 11.5, ORANGE, "start"))
    b.append(label(30, 610, "Every arrow out of a blue box is reproducible from the registry with one command; the holdout is guarded in code and was used exactly once.", 11, MUTED, "start"))
    (OUT / "pipeline_overview.svg").write_text(svg_wrap(W, H, "\n".join(b)), encoding="utf-8")


def diagram_validation():
    """How the 1,460 rows are used."""
    W, H = 1000, 430
    b = []
    b.append(label(30, 32, "How the 1,460 labelled houses are used", 15, INK, "start", "700"))
    # full bar
    dev_w = 1241 / 1460 * 900
    b.append(f"<rect x='30' y='55' width='{dev_w:.0f}' height='44' rx='6' fill='#f3f7fd' stroke='{BLUE}'/>")
    b.append(f"<rect x='{30 + dev_w:.0f}' y='55' width='{900 - dev_w:.0f}' height='44' rx='6' fill='#fdf1ec' stroke='{ORANGE}'/>")
    b.append(label(30 + dev_w / 2, 82, "development: 1,241 rows (all experiments, tuning, selection)", 12.5, INK, weight="700"))
    b.append(label(30 + dev_w + (900 - dev_w) / 2, 76, "holdout", 12, INK, weight="700"))
    b.append(label(30 + dev_w + (900 - dev_w) / 2, 92, "219 rows", 11, INK2))
    b.append(label(940, 82, "", 1))
    # folds seed 42
    b.append(label(30, 135, "Selection CV: KFold(5, shuffle, seed 42) - used to choose features, hyper-parameters, ensembles", 12, BLUE, "start", "700"))
    fw = dev_w / 5
    for r in range(5):
        y = 148 + r * 26
        for k in range(5):
            val = (k == r)
            fill = BLUE if val else "#dbe8f8"
            b.append(f"<rect x='{30 + k * fw + 1:.0f}' y='{y}' width='{fw - 3:.0f}' height='20' rx='3' fill='{fill}'/>")
        b.append(label(40 + dev_w, y + 15, f"fold {r + 1}", 10.5, INK2, "start"))
    b.append(label(30, 298, "dark = validation part of the fold (scored), light = training part (fits imputers, skew choice, encoder, scaler, model, ensemble weights)", 10.5, INK2, "start"))
    # fresh folds + holdout
    b.append(label(30, 335, "Finalist check: RepeatedKFold(5 x 3, seed 7) - 15 new partitions, used once to pick among 8 finalists", 12, ORANGE, "start", "700"))
    b.append(label(30, 362, "Holdout: scored ONCE after the model was frozen in Git; score_holdout_once() refuses a second call.", 12, ORANGE, "start", "700"))
    b.append(label(30, 395, "Why three layers? A score you used to choose is optimistic (measured here: +0.003 to +0.004).", 11.5, INK2, "start"))
    b.append(label(30, 414, "Fresh folds correct that; the holdout is data no decision ever touched.", 11.5, INK2, "start"))
    (OUT / "validation_protocol.svg").write_text(svg_wrap(W, H, "\n".join(b)), encoding="utf-8")


def diagram_nested():
    """Nested stacking inside one outer fold."""
    W, H = 1000, 380
    b = []
    b.append(label(30, 30, "Nested stacking: how the meta-learner avoids seeing its own validation rows", 15, INK, "start", "700"))
    b.append(box(30, 55, 260, 80, "Outer training fold", ["~993 rows (4 of 5 outer folds)"], "#f3f7fd", BLUE))
    b.append(box(30, 255, 260, 80, "Outer validation fold", ["~248 rows: only predicted, never fitted"], "#fdf1ec", ORANGE))
    b.append(box(350, 45, 290, 100, "Inner 5-fold CV", ["fit Lasso and CatBoost on inner-train", "predict inner-validation", "→ out-of-fold predictions for every row"], "#f3f7fd", BLUE))
    b.append(box(700, 45, 270, 100, "Meta-learner (Ridge / NNLS)", ["learns weights from OOF predictions", "(never from in-sample fits)"], "#f3f7fd", BLUE))
    b.append(box(700, 185, 270, 70, "Refit bases on all outer-train rows", [], "#f3f7fd", BLUE, fs=12))
    b.append(box(350, 255, 290, 80, "Score", ["meta(bases(outer validation))", "→ one honest fold score"], "#fdf1ec", ORANGE))
    b.append(arrow(290, 95, 350, 95))
    b.append(arrow(640, 95, 700, 95))
    b.append(arrow(835, 145, 835, 185))
    b.append(arrow(835, 255, 640, 295))
    b.append(arrow(290, 295, 350, 295))
    b.append(label(30, 365, "In code this is automatic: the ensemble is an ordinary estimator whose fit() runs the inner CV, so the outer cross_validate() nests it.", 11, MUTED, "start"))
    (OUT / "nested_stacking.svg").write_text(svg_wrap(W, H, "\n".join(b)), encoding="utf-8")


def diagram_leakage():
    """Inside vs outside the fold."""
    W, H = 1000, 300
    b = []
    b.append(label(30, 30, "The leakage rule: which steps may run before cross-validation?", 15, INK, "start", "700"))
    b.append(box(30, 50, 450, 210, "Safe OUTSIDE the fold", [
        "output for a row depends on that row only",
        "",
        "TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF",
        "OverallQual x GrLivArea",
        "GarageType missing & GarageArea = 0 → 'None'",
        "quality labels → ordered numbers",
        "log1p of the target",
    ], "#eef8f3", AQUA))
    b.append(box(520, 50, 450, 210, "Must be INSIDE the fold", [
        "output depends on statistics of other rows",
        "",
        "LotFrontage ← neighbourhood median",
        "which columns are 'skewed' (> 0.75)",
        "one-hot categories, scaler mean / sd",
        "outlier removal (training side only)",
        "ensemble weights, meta-learner, hyper-parameters",
    ], "#fdf1ec", ORANGE))
    b.append(label(30, 288, "Implemented as a scikit-learn Pipeline cloned and refitted per fold: leakage safety is a property of the structure, not of discipline.", 11, MUTED, "start"))
    (OUT / "leakage_rule.svg").write_text(svg_wrap(W, H, "\n".join(b)), encoding="utf-8")


if __name__ == "__main__":
    reg = registry()
    fig_ladder(reg)
    fig_finalists()
    fig_outliers()
    fig_deciles()
    fig_optuna()
    diagram_pipeline()
    diagram_validation()
    diagram_nested()
    diagram_leakage()
    print("figures written to", OUT)
