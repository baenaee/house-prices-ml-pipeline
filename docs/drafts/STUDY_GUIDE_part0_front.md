# Study guide: Kaggle House Prices, from a cleaning notebook to a frozen, audited model

This guide explains everything the project did, why it did it, and how the code does it, in enough detail for you to defend each decision without the repository open. It is long on purpose. Read it in order once; afterwards the self-check questions at the end of each part are the fastest way to revise.

## How the guide is organised

| Part | Topic | Written by |
|---|---|---|
| 1 | The data, the 17 cleaning rules, fitted statistics and leakage, the engineered feature groups, outliers, test-only anomalies | feature-researcher |
| 2a | Validation methodology: the metric, K-fold CV, selection bias, a leakage taxonomy, reproducibility, what the audits found | feature-researcher |
| 2b | Linear models: OLS, Ridge, Lasso, ElasticNet, skew correction, scalers, the grid results | feature-researcher |
| 2c | Boosted trees and Optuna: trees, gradient boosting, XGBoost, LightGBM, CatBoost, TPE, the staged tuning budget, Stage 1 and Stage 2 | model-researcher |
| 3 | Ensembles: why averaging helps, learned blend weights, nested stacking, diversity, the final ensemble results, finalists on fresh folds | ensemble-researcher |
| 4 | From finalists to a frozen model: the decision rule, the single holdout evaluation, the final fit and the submission file | director |
| 5 | What the final model learned and where it fails (interpretability and error analysis) | model-researcher |
| 6 | The Python implementation: how the package fits together and how to run everything | director |
| 7 | Defending the project: the decision list, what was rejected, the limitations, and a one-page mental model | director |

The parts were written by different agents at different moments of the campaign and were merged with light editing. Where an early part describes a state that later changed (for example "Stage 1" results in part 3), the text says so and points to the section with the final numbers. The authoritative record of every number is the experiment registry (`experiments/registry.jsonl`) and the decision log (`docs/EXPERIMENT_HISTORY.md`, decisions D1 to D22).

## The whole project in one page

**Problem.** Predict the sale price of 1,459 houses in Ames, Iowa, from 79 descriptive columns, learning from 1,460 houses with known prices. Kaggle scores the root mean squared error of the logarithm of the price (RMSLE). The project predicts `log1p(SalePrice)` and reports RMSE on that scale, which is the same number.

**Protocol.** 219 training houses (15 %, stratified by price decile) were set aside once as a holdout and locked in code. Every experiment used the other 1,241 "development" rows with a shuffled 5-fold cross-validation (seed 42). Everything that learns from data (imputation statistics, the choice of skewed columns, encoders, scalers, models, ensemble weights) is fitted inside each training fold. Candidates for the final model were re-scored on 15 new folds (3 repeats of 5 folds, seed 7) because scores on the folds used for choosing are optimistic. Then one procedure was frozen, approved by an independent auditor, and scored on the holdout exactly once.

**The ladder of results** (RMSE on the log scale; lower is better):

| step | what changed | selection folds | id |
|---|---|---|---|
| predict the mean for everybody | nothing learned | 0.3987 | E001 |
| ordinary least squares on numeric columns | first real model | 0.1599 | E002 |
| Ridge on one-hot categories | categories + shrinkage | 0.1409 | E003 |
| Ridge + engineered features + in-fold skew correction | feature engineering | 0.1304 | E041 |
| Lasso, tuned alpha | sparsity | 0.1255 | E056 |
| CatBoost, 25 Optuna trials | boosted trees | 0.1195 | E060 |
| CatBoost, 75 Optuna trials | more tuning | 0.1181 | E066 |
| average of Lasso and CatBoost | ensemble | 0.1173 | E068 |

**The honest numbers.** On the fresh folds the frozen model, single CatBoost E066, scores 0.1217 and the Lasso + CatBoost average 0.1207; the difference (0.0010) is a third of its standard error. On the single-use holdout the frozen model scores 0.1065. The project's claim is: expected RMSLE about 0.12, with the holdout as a consistency check and not as a better estimate.

**Three results that go against common practice in public notebooks.**
1. Removing the famous outliers made every model worse once the removal was restricted to training folds (part 1, chapter 5).
2. The 75-trial tuned CatBoost was not significantly improved by any ensemble; learned blend weights and stacking never beat a plain 50/50 average (part 3).
3. Every score chosen on the selection folds dropped by 0.003 to 0.004 when measured on new folds; an analysis that reports the selection score is reporting a number that is too good (parts 2a and 4).
