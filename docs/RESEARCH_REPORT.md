# Research report: Kaggle House Prices (Ames)

Final version, 2026-11-17. Every number comes from `experiments/registry.jsonl` (83 experiments, E001–E083), `experiments/holdout_usage.log` or the files named next to it. Scores are RMSE on `log1p(SalePrice)`, the competition's RMSLE; lower is better.

## 1. Summary

- **Final model:** a single CatBoost regressor on one-hot inputs with six tuned hyper-parameters (registry id E066, configuration hash `be58905cb40a`), trained on all 1,460 training rows, no outlier removal.
- **Expected error:** about **0.12**. The cross-validated estimate on fresh folds (not used for tuning or for generating candidates; used once, under a pre-registered rule, to choose among eight finalists) is 0.1217 (15 folds, fold standard deviation 0.013).
- **Holdout:** 0.1065 (95 % bootstrap interval 0.092–0.121) on 219 houses, evaluated once after the procedure was frozen and approved by the independent auditor. This is inside the range the auditor wrote down beforehand (0.100–0.145) and on its favourable side; it is a consistency check, not a better estimate (section 8).
- **Against the baselines:** predicting the mean scores 0.399, a numeric-only linear regression 0.160. On the selection folds the final model removes about 70 % of the mean baseline's error (0.1181 against 0.3987).
- **The best alternative** is the equal-weight average of Lasso and CatBoost: 0.1207 on fresh folds, 0.0010 better than the frozen model, a difference of one third of a standard error. It was not chosen, by a rule fixed before the comparison was read (section 7).
- **Kaggle:** `outputs/submissions/E066_all.csv` was submitted once after the freeze. Kaggle's score on the whole test set (1,459 houses) is **0.12271**, 0.001 from the fresh-fold estimate; rank 1,225 of 4,717 on 2026-11-26 (section 10).

## 2. Problem, data and protocol

The task is to predict the sale price of 1,459 houses in Ames, Iowa, from 79 descriptive columns, given 1,460 houses with prices. The project started from an existing exploratory notebook with 17 documented cleaning rules and no model.

Protocol, fixed before the first experiment and enforced in code:

| element | choice |
|---|---|
| target and metric | `log1p(SalePrice)`; RMSE on that scale |
| holdout | 219 rows (15 %), stratified by price decile, seed 42, Ids in `experiments/holdout_ids.json`; scored once through `validation.score_holdout_once` |
| development set | the other 1,241 rows |
| selection folds | `KFold(5, shuffle=True, random_state=42)` |
| finalist folds | `RepeatedKFold(5, 3, random_state=7)`; no fold identical to a selection fold (validation overlap up to 25 %, 20 % expected by chance; audit F20) |
| preprocessing | every fitted step inside the pipeline that cross-validation refits per fold |
| outlier filters | applied after the split, to training rows only |
| stacking and blending | nested: weights and meta-learner learned from inner out-of-fold predictions inside each outer-training fold |
| evidence | `python -m houseprices.evaluate --experiment <id>` must reproduce the registered fold scores |
| decision threshold | differences below 0.003 are not treated as wins (measured selection optimism) |

## 3. Data preparation

**Cleaning.** The 17 rules of the original notebook were ported to pure functions (`src/houseprices/cleaning.py`) and every count the notebook printed is asserted by a test. They are deterministic and row-wise or dictionary-based (for example "a missing garage type with zero garage area means no garage"); none uses the target or a statistic fitted on the data. One cell of the original notebook (rule C01) was missing from every copy and was reconstructed from the cleaning log; it is labelled as a reconstruction. The original notebook and CSV files are byte-identical to their state before the project (SHA-256 in `docs/AUDIT_2026-11-11.md`).

**Features.** Seven named groups in `features.py`, all row-wise arithmetic: `base` (the cleaned columns), `areas` (TotalSF and related sums), `baths`, `ages`, `quality_interactions` (OverallQual × GrLivArea, OverallQual × TotalSF), `ordinal_quality` (quality labels as ordered numbers), `anomaly_flags`.

**In-fold preprocessing.** LotFrontage imputed with the neighbourhood median of the training fold; for linear models, `log1p` of numeric columns whose training-fold skewness exceeds 0.75, then scaling; one-hot encoding with unknown categories ignored.

## 4. Experiments and results

### 4.1 Baselines (E001–E004)

| model | CV RMSE | fold sd |
|---|---|---|
| mean of log price | 0.3987 | 0.027 |
| OLS on numeric columns | 0.1599 | 0.035 |
| Ridge (alpha 10), one-hot | 0.1409 | 0.032 |
| XGBoost defaults | 0.1487 | 0.019 |

### 4.2 Feature engineering (E005–E048)

Each group was added to, and removed from, the base set for Ridge, ElasticNet and XGBoost. Only `quality_interactions` mattered for every model (removing it cost 0.0025–0.009). The in-fold skew correction was the largest single gain for linear models (about 0.009: Ridge 0.1387 → 0.1304, ElasticNet 0.1350 → 0.1256). The other groups moved scores by 0.003 or less. LotFrontage imputation method (neighbourhood median, global median, KNN) made no difference beyond 0.002. The full set was kept (D11) because pruning at untuned hyper-parameters would fit noise.

### 4.3 Outlier policy (E049–E054)

Removing the two houses usually deleted in public solutions (Ids 524, 1299), or all four above 4,000 sq ft, from the training side of each fold while validating on complete folds made all three models worse: by 0.004–0.009 for the linear models and 0.001–0.005 for XGBoost, three of six cells beyond one paired standard error. All rows are kept (D15). The gain commonly reported for outlier removal on this data comes from also removing those houses from the validation data.

### 4.4 Linear models (E055–E057)

136 grid points on the selection folds: Ridge 0.1303, Lasso 0.1255 (alpha 5.6e-4), ElasticNet 0.1256. The L1 models beat Ridge in every fold. Scaler choice was irrelevant (differences up to 0.0001).

### 4.5 Boosted trees (E058–E060, E065–E067)

Optuna TPE, objective = mean 5-fold CV RMSE, staged budget.

| booster | Stage 1, 25 trials | Stage 2, 75 trials | minutes (both stages) |
|---|---|---|---|
| XGBoost | 0.1231 (E058) | not run (flat since trial 4, gap > 0.003) | 2.2 |
| LightGBM | 0.1244 (E059) | 0.1217 (E067) | 4.9 |
| CatBoost, one-hot | 0.1195 (E060) | 0.1181 (E066) | 36.9 |
| CatBoost, native categoricals | 0.1221 after 5 of 25 trials, time-capped (E065) | not run | 28.0 |

Tuning used 72 of the 180 minutes allowed. Stage 2 gains (0.0014 and 0.0027) are inside the 0.003 band on the selection folds; on fresh folds the CatBoost gain is 0.0016, the same in all three repeats. Winning configurations are shallow, slow and subsampled (CatBoost: depth 6, learning rate 0.017, 2,000 trees, 59 % row subsample). 58 of the 75 CatBoost trials are within 0.003 of the best: the model is on a plateau. Native categorical handling was worse than one-hot in all five trials that share hyper-parameters, by 0.0016–0.0028, at 15–24 times the cost; that study was stopped and is reported as partial.

### 4.6 Ensembles (E061–E064, E068–E075)

Residual correlations between tuned bases: 0.970–0.997 among linear models, 0.939–0.972 among boosters, 0.825–0.860 between the families. The only real diversity is between a linear model and a tree model.

| ensemble (final bases) | CV RMSE | vs CatBoost E066 (paired, 5 folds) |
|---|---|---|
| equal-weight average, Lasso + CatBoost (E068) | 0.1173 | −0.0007 ± 0.0019 |
| nested stack, Lasso + CatBoost (E072) | 0.1175 | −0.0006 ± 0.0017 |
| NNLS blend, Lasso + CatBoost (E069) | 0.1180 | −0.0001 ± 0.0017 |
| nested stack, six bases (E073) | 0.1185 | +0.0004 ± 0.0022 |
| nested stack, three boosters (E075) | 0.1186 | +0.0005 ± 0.0012 |
| NNLS blend, three boosters (E071) | 0.1187 | +0.0007 ± 0.0009 |
| NNLS blend, six bases (E070) | 0.1189 | +0.0009 ± 0.0023 |

No ensemble differs from single CatBoost by more than one paired standard error. Learned weights never beat the plain average. As CatBoost improved (Stage 1 to Stage 2), the benefit of averaging with Lasso halved (−0.0014 to −0.0007): with a residual correlation of 0.856 and an error ratio of 0.127 to 0.119 the two-model variance formula predicts a gain of about 0.0004, and the break-even correlation is 0.869.

## 5. Selection optimism, measured

| measurement | optimism |
|---|---|
| Ridge grid of 15 points, re-scored on five fresh partitions (audit round 4) | +0.002 |
| Lasso best of grid, seed 42 → seed 7 | +0.0027 |
| eight finalists, seed 42 → seed 7 | +0.0027 to +0.0044, mean +0.0036 |

Every reported "selection-fold" number in this project is therefore too good by about 0.003–0.004. The fresh-fold numbers are the ones to quote.

## 6. Finalists on fresh folds (E076–E083)

`RepeatedKFold(5×3, seed 7)`; differences are paired against single CatBoost E066 on the same 15 folds. The standard error is sd/√5, following audit finding F30 (the three repeats re-partition the same rows and are not independent; the runner's sd/√15 is too small; the Nadeau–Bengio correction gives 0.0038 for the first row).

| model | selection folds | fresh folds | diff vs E066 | SE | folds won |
|---|---|---|---|---|---|
| average Lasso + CatBoost (E068) | 0.1173 | 0.1207 | −0.0010 | 0.0031 | 9 / 15 |
| average, Stage 1 bases (E061) | 0.1181 | 0.1215 | −0.0002 | 0.0036 | 9 / 15 |
| nested stack Lasso + CatBoost (E072) | 0.1175 | 0.1216 | −0.0000 | 0.0028 | 10 / 15 |
| **CatBoost, 75 trials (E066)** | 0.1181 | **0.1217** | reference | | |
| NNLS blend Lasso + CatBoost (E069) | 0.1180 | 0.1224 | +0.0007 | 0.0026 | 9 / 15 |
| CatBoost, 25 trials (E060) | 0.1195 | 0.1233 | +0.0016 | 0.0011 | 5 / 15 |
| LightGBM, 75 trials (E067) | 0.1217 | 0.1252 | +0.0036 | 0.0016 | 4 / 15 |
| Lasso (E056) | 0.1255 | 0.1282 | +0.0065 | 0.0066 | 6 / 15 |

## 7. Final model selection (ruling D22)

The auditor recorded a decision rule before reading the finalist records: keep single CatBoost unless an ensemble is at least 0.0010 better on fresh folds, by more than one standard error, winning at least 10 of 15 folds with all three repeat means negative. The average met the first and last conditions and failed the other two.

Single CatBoost E066 was frozen. Reasons: no ensemble shows a credible improvement; one pipeline is simpler to explain, audit and reproduce; and a linear member extrapolates on houses outside the training range (test Id 2550) where a tree does not. This is a parsimony ruling. The data do not show the average to be worse, and cannot exclude a true difference of about 0.002 in either direction.

Frozen specification: `houseprices.models:build_catboost(learning_rate=0.01669422348518078, iterations=2000, depth=6, l2_leaf_reg=1.413619495827429, subsample=0.5869892858785986, rsm=0.8128738808596279)`, feature set `base+areas+baths+ages+quality_interactions+ordinal_quality`, training-row filter `keep_all`, target `log1p(SalePrice)`. Freeze commit `95c3e86`; auditor approval in `docs/VALIDATION_AUDIT.md` §9.9.

## 8. The holdout evaluation

One scoring call to `validation.score_holdout_once` on 2026-11-17 (the script then re-predicted with the same fitted model to save the residuals, which reproduce the logged score; audit note F35), model fitted on the 1,241 development rows (commit `5beae61`), logged in `experiments/holdout_usage.log`.

| quantity | value |
|---|---|
| holdout RMSE (219 rows) | **0.10645** |
| 95 % bootstrap interval of that RMSE (auditor, 20,000 resamples of the 219 squared residuals) | 0.092 – 0.121 |
| range pre-registered as consistent with CV | 0.100 – 0.145 |
| fresh-fold CV estimate | 0.1217 |
| mean signed residual / mean absolute residual | +0.004 / 0.076 |
| residuals above 0.3 in absolute value | 3 of 219 (largest 0.41) |
| median absolute percentage error in dollars | 5.3 % |

The holdout score is lower than the CV estimate. Three reasons make that unsurprising, and none makes 0.106 the expected error: the holdout contains no house above 4,000 sq ft (its largest is 3,627), while the four such houses, all development rows, carry 8.5 % of the out-of-fold squared error and raise the pooled development RMSE by 0.005 (the five fresh folds containing none of them average 0.114, the ten containing at least one average 0.126); an RMSE over 219 rows is a noisy quantity (bootstrap standard deviation 0.007); and the holdout model was fitted on 1,241 rows where CV models see about 993 (worth roughly 0.001). The fresh-fold CV estimate, 0.1217, lies just above the upper end of the holdout's bootstrap interval; compared like with like (development folds without the four large houses) the gap disappears. The holdout confirms that the cross-validated estimate was not inflated. It cannot rank the frozen model against the average, and it was used for no decision. Nothing was changed after it was scored.

## 9. What the model relies on and where it fails

Source: `experiments/interpretability/run_interpretability.py --experiment E066`; tables in `outputs/interpretability/`, figures in `outputs/figures/`; full text in part 5 of the study guide. Out-of-fold predictions on the selection folds (pooled RMSE 0.1190).

**What it uses.** Permutation importance measured on validation folds (increase in RMSE when a raw column is shuffled): `OverallQual_x_TotalSF` +0.064, `OverallQual_x_GrLivArea` +0.021, `TotalSF` +0.017, then `OverallCond`, `MSZoning`, `OverallQual`, `LotArea`, `Neighborhood`, `TotalBath` at +0.005 to +0.008. Only 32 of 99 raw columns raise the error by more than 0.001. SHAP on the fitted model gives the same first three. These columns are strongly related, so the correct reading is "size times quality drives the prediction", not a ranking among them.

**Where it fails.**

| segment | RMSE | mean residual (actual − predicted) |
|---|---|---|
| cheapest price decile (≤ 106,500) | 0.215 | −0.097 (over-predicted) |
| deciles 2 to 9 | 0.076 – 0.106 | near zero |
| most expensive decile (≥ 278,000) | 0.153 | +0.072 (under-predicted) |
| living area 1,500–1,999 sq ft (393 houses) | 0.093 | |
| living area 3,000–3,999 sq ft (12 houses) | 0.201 | +0.101 |
| living area ≥ 4,000 sq ft (4 houses) | 0.611 | |
| worst neighbourhoods: IDOTRR (n=29), Edwards (n=86), NoRidge (n=34) | 0.239, 0.165, 0.162 | |
| best neighbourhoods: CollgCr (n=130), BrDale, Gilbert | 0.069, 0.066, 0.063 | |

The error is concentrated: the worst 5 rows carry 13.6 % of the squared error and the worst 15 carry 31.1 %, while only 6.6 % of houses have an absolute residual above 0.2. Without the four houses above 4,000 sq ft the pooled RMSE falls from 0.1190 to 0.1140. Two of those four (partial sales in Edwards) are over-predicted by a factor of about 1.8–1.95 and two (NoRidge) under-predicted by about 1.7–1.9, so the training data give opposite signals for the one comparable test house (Id 2550). The holdout residuals show the same sign pattern by price tercile (−0.023, +0.001, +0.035) with smaller magnitude.

## 10. Final fit and submission file

`train --experiment E066 --rows all` (1,460 rows) and `predict` produced `outputs/submissions/E066_all.csv`: 1,459 rows, columns `Id,SalePrice`, Id order equal to `sample_submission.csv`, prices from 41,901 to 524,842 (median 156,611; training median 163,000, maximum 755,000). The three test-only anomalies were checked by hand: Id 2593 (garage year 2207) 256,431; Id 1877 212,580; Id 2550 (5,095 sq ft, Edwards, partial sale) 268,438. The file was submitted to Kaggle once, after the freeze and the holdout evaluation, and scored **0.12271** on the whole test set (the leaderboard is computed on all 1,459 test houses), rank 1,225 of 4,717 teams on 2026-11-26 (top 26 %). This is the only evaluation on data from outside the training file, and it agrees with the fresh-fold estimate (0.1217) within 0.001, well inside the fold standard deviation (0.013). The score was recorded after the fact and used for no decision. A public Kaggle notebook that rebuilds the model from the GitHub repository produced an identical submission and score. The leaderboard's top entries score 0.00000, possible only by copying the publicly available true prices, so the rank understates the model.

## 11. Hypotheses, as stated before the evidence

| hypothesis | outcome |
|---|---|
| H1: Stage 2 improves CatBoost by less than the 0.003 noise band | confirmed (0.0014 on selection folds, 0.0016 on fresh folds) |
| H2: the equal-weight Lasso + CatBoost average remains the best procedure on fresh folds and learned weights do not beat it | point estimate yes; not distinguishable from single CatBoost |
| H3: native categoricals are not worth their cost on 1,241 rows | supported by five paired trials; study incomplete |
| H4: the holdout lands within about 0.01 of the fresh-fold mean | refuted: 0.015 away, on the favourable side |

## 12. Limitations and uncertainties

1. The top two procedures are statistically indistinguishable; the choice between them is a parsimony judgement.
2. All model selection used the same 1,241 rows. Fresh folds correct the reported number but are not new data; only the 219-row holdout is, and its sampling error is about 0.01.
3. Both Stage 2 optima lie on the upper bound of the tree-count range and LightGBM's `min_child_samples` on its floor; the search box may exclude slightly better settings.
4. The error is concentrated in a few atypical houses and at the price extremes (section 9). Test Id 2550 is such a house; its prediction is uncertain in both directions and CV cannot settle it.
5. The anomaly flags are constant in the training data, so no model can learn from them; the three test-only anomalies are handled by feature definitions and checked by hand, not validated.
6. The data cover one city over 2006–2010; validation was random, not time-ordered.
7. The CatBoost-native comparison is partial (5 of 25 trials).
8. Tooling: registry without locking (single writer by process), joblib artefacts, an Optuna study that is not exactly re-creatable after the Stage 2 resume. Registered results do not depend on any of these.
9. There is a single leaderboard score (0.12271). One score on 1,459 houses confirms the estimate; it cannot rank close alternatives, and it was not used to.

## 13. Reproduction

```powershell
conda env create -f environment.yml; conda activate houseprices; pip install -e .
python -m pytest tests
python -m houseprices.evaluate --experiment E066     # selection-fold score of the frozen model
python -m houseprices.evaluate --experiment E076     # its fresh-fold score
python -m houseprices.train --experiment E066 --rows all
python -m houseprices.predict --model outputs/models/E066_all.joblib
```

Related documents: `docs/EXPERIMENT_HISTORY.md` (chronology and rulings D1–D22), `docs/VALIDATION_AUDIT.md` (findings F1 onward and sign-offs), `docs/STUDY_GUIDE.md` (the methods explained), `docs/CHATGPT_AUDIT_PACKET.md` (claims and the commands that check them).
