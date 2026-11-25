"""Write the public Kaggle showcase notebook (portfolio/kaggle/house-prices-leakage-safe-pipeline.ipynb).

The notebook clones the GitHub repository, points it at the competition data attached on Kaggle,
reproduces the frozen model's registered CV score, refits it on all rows and writes submission.csv.
It needs "Internet on" in the Kaggle notebook settings (to clone the repo and pip-install pinned versions).

    python portfolio/build/make_kaggle_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "portfolio" / "kaggle" / "house-prices-leakage-safe-pipeline.ipynb"
REPO = "https://github.com/baenaee/house-prices-ml-pipeline"
RAW = "https://raw.githubusercontent.com/baenaee/house-prices-ml-pipeline/main"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md(f"""# House Prices: a leakage-safe, reproducible pipeline (RMSLE ≈ 0.12, honestly measured)

This notebook is the Kaggle companion of **[{REPO.split('github.com/')[1]}]({REPO})**. It doesn't chase the leaderboard; it shows a validation protocol whose numbers you can trust, and lets you reproduce them in a few minutes.

**What's in the repository:** a tested Python package, 83 logged experiments (each reproducible with one command), a [technical report]({REPO}/blob/main/portfolio/HousePrices_Technical_Report.pdf), [slides]({REPO}/blob/main/portfolio/HousePrices_Slides.pdf) and a [model card]({REPO}/blob/main/docs/MODEL_CARD.md).

**What this notebook does:**
1. clones the repository and points it at the competition data attached here (the data are not redistributed on GitHub, per the competition rules);
2. reproduces the cross-validation score registered for the final model (CatBoost, experiment `E066`) from its stored configuration and seed;
3. refits that model on all 1,460 training rows and writes `submission.csv`.

> Settings needed: **Internet on** (to clone the repo and install the pinned library versions). Runtime ≈ 5 minutes on CPU.

![Project summary]({RAW}/portfolio/figures/social_preview.png)"""),

    md("""## The protocol in one paragraph

A 15 % holdout (219 houses, stratified by price decile) was locked before any modelling. Every choice was made with 5-fold CV on the other 1,241 rows, with **everything that learns from data fitted inside each training fold** (imputation, which columns to log-transform, encoding, scaling, the model, ensemble weights via nested stacking). Because the best of many configurations is optimistic on the folds it was chosen on, the eight finalists were re-scored on 15 fresh folds (they all got worse by 0.003–0.004). The final model was chosen by a rule written before the comparison was read, frozen in Git, and scored on the holdout exactly once.

| estimate | RMSLE |
|---|---|
| predict the mean | 0.399 |
| best linear model (Lasso), fresh folds | 0.128 |
| **final CatBoost, fresh folds (the number to quote)** | **0.1217** |
| Lasso + CatBoost average, fresh folds (a statistical tie, not chosen) | 0.1207 |
| final CatBoost, single-use holdout | 0.1065 |
| final CatBoost, this competition's test set (one submission) | 0.1227 |"""),

    md("## 1. Get the code and point it at the competition data"),
    code(f'''import glob, os, shutil, subprocess, sys
from pathlib import Path

REPO_URL = "{REPO}.git"
ON_KAGGLE = Path("/kaggle/input").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("HP_WORK_DIR", ".")).resolve()
REPO = WORK / "house-prices-ml-pipeline"

def run(cmd, **kw):
    """Run a command, show its output in the notebook, stop the notebook if it fails."""
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(r.stdout[-6000:] + r.stderr[-3000:])
    r.check_returncode()

if not REPO.exists():
    run(["git", "-c", "core.longpaths=true", "clone", "--depth", "1", REPO_URL, str(REPO)])

# Competition data: attached under /kaggle/input on Kaggle (HP_DATA_DIR elsewhere)
if ON_KAGGLE:
    data_dir = Path(sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))[0]).parent
else:
    data_dir = Path(os.environ["HP_DATA_DIR"])
for name in ["train.csv", "test.csv", "sample_submission.csv", "data_description.txt"]:
    shutil.copy(data_dir / name, REPO / name)

# The same check the repository's download script runs: SHA-256 equal to the files every experiment used
run([sys.executable, "scripts/download_data.py"], cwd=REPO)'''),

    md("Install the exact library versions the experiments were run with (pandas 2.3.3, scikit-learn 1.9.1, CatBoost 1.2.10, …). Only needed on Kaggle; it takes a minute or two."),
    code('''if ON_KAGGLE:
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-ci.txt"], cwd=REPO)

ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"))'''),

    md("""## 2. Reproduce the registered score of the final model

`houseprices.evaluate` rebuilds the pipeline from the experiment registry (builder + JSON arguments + feature set), rebuilds the folds from the stored seed, re-scores them and compares fold by fold with the stored scores. A registered number only counts in this project if this returns `reproduces_registered_scores: true`."""),
    code('''run([sys.executable, "-m", "houseprices.evaluate", "--experiment", "E066"], cwd=REPO, env=ENV)'''),

    md("""What that pipeline is: the frozen configuration as stored in the registry. Note that preprocessing is part of the estimator, which is why cross-validation refits it on each training fold."""),
    code('''import json
reg = {json.loads(l)["id"]: json.loads(l) for l in open(REPO / "experiments" / "registry.jsonl", encoding="utf-8")}
e = reg["E066"]
print("builder:      ", e["builder"])
print("arguments:    ", json.dumps(e["builder_kwargs"], indent=2))
print("feature set:  ", e["feature_set"])
print("row filter:   ", e["train_row_filter"], "(no outlier removal: removing them made every model worse)")
print("fold scores:  ", e["fold_scores"], "-> mean", round(e["cv_mean"], 4))
print("fresh folds:  ", round(reg["E076"]["cv_mean"], 4), "(RepeatedKFold 5x3, seed 7)")'''),

    md("""## 3. Refit on all 1,460 rows and write the submission

`train --rows all` is refused by the code until the holdout has been scored (it has, once; see `experiments/holdout_usage.log`). `predict` refuses to write a file whose Ids differ from the sample submission or whose prices are not finite and positive."""),
    code('''run([sys.executable, "-m", "houseprices.train", "--experiment", "E066", "--rows", "all"], cwd=REPO, env=ENV)
run([sys.executable, "-m", "houseprices.predict", "--model", "outputs/models/E066_all.joblib",
     "--out", str(WORK / "submission.csv")], cwd=REPO, env=ENV)

import pandas as pd
sub = pd.read_csv(WORK / "submission.csv")
ref = pd.read_csv(REPO / "outputs" / "submissions" / "E066_all.csv")   # the file produced in the original run
print(sub.shape, "| price range", int(sub.SalePrice.min()), "-", int(sub.SalePrice.max()))
print("max relative difference vs the repository's submission:",
      f"{(sub.SalePrice / ref.SalePrice - 1).abs().max():.2e}")
sub.head()'''),

    md(f"""## What the project found

1. **Removing the "famous outliers" made every model worse** once they were removed from training folds only (validation folds keep every house). The usual gain comes from also deleting them from the validation data.
2. **Selection optimism is real and measurable**: scores used for choosing were 0.003–0.004 too good.
3. **Ensembles were within noise**: the best (a plain Lasso + CatBoost average) was 0.001 better than CatBoost alone, a third of a standard error; learned blend weights and stacking never beat the plain average. The simpler model was kept by a pre-registered rule.
4. **Errors concentrate**: 15 of 1,241 houses carry 31 % of the squared error; cheap houses are over-predicted and expensive ones under-predicted.

![Selection vs fresh folds]({RAW}/portfolio/figures/finalists_fresh_folds.png)

Full details, every decision with its evidence, and the independent code audit are in the [repository]({REPO}). Feedback welcome!

*Built with Claude Code agent teams under the author's methodology and review.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT.relative_to(ROOT))
