# Study guide, part 7: defending the project

This part is the revision sheet. It lists every binding decision with the one-sentence reason you should be able to give, what was tried and rejected, the limitations you should volunteer before an interviewer finds them, and the questions most likely to be asked.

## 1. The decisions, with the sentence that defends each

| # | Decision | The sentence |
|---|---|---|
| D1 | Target `log1p(SalePrice)`, metric RMSE on that scale | It is the competition metric, and errors in log price are relative errors, which is how house prices are judged. |
| D2 | 5-fold shuffled CV (seed 42) for selection; 5x3 repeated CV for finalists | With 1,241 rows, 5 folds keep about 993 rows for training; repeats measure how much a score depends on the split. |
| D3, D15 | Outlier filters act on training folds only; final ruling: keep all rows | Deleting hard houses from validation makes the score better without making the model better; measured honestly, removal made every model worse. |
| D4 | 15 % holdout, stratified by price decile, single use | An estimate that no decision has touched; one use, because every reuse turns it into a tuning set. |
| D5, D6 | Rule B04 as a condition, rule C01 reconstructed and labelled as such | A rule tied to a row Id does not generalise; a lost cell is documented as reconstructed, not claimed as recovered. |
| D7 | pandas pinned to 2.3.3 | The original notebook's behaviour depends on the string dtype that pandas 3 changes. |
| D8 | CV raises on non-finite predictions and empty folds | A NaN score would be stored silently and sorted last. |
| D9 | `Id` travels with the features and is dropped inside every builder | Filters and the submission need it; no model may see it. |
| D11 | Feature set "all" (base + areas + baths + ages + quality interactions + ordinal quality) | Only the quality interactions clearly helped; pruning the others at default hyper-parameters would be fitting to noise. |
| D12 | Skew correction for linear models only | Worth about 0.009 for them; trees are invariant to monotone transforms. |
| D13 | LotFrontage: neighbourhood median, fitted in the fold | No alternative differed beyond noise; keep the documented default. |
| D14 | One-standard-error verdicts on 5 folds are provisional | Differences near 0.001 are noise at 5 folds. |
| D16 | Lasso and ElasticNet are the linear finalists | The L1 models beat Ridge by 0.005 in every fold. |
| D17 | Finalists are scored with seed 7 | Seed 42 folds were used for choosing; reusing them repeats the optimism. |
| D18, D20 | Test-only anomalies documented; year clipper clips the upper bound only | Problems invisible to CV must be reported; clipping the lower bound would have altered two real 1890s garages. |
| D19, D21 | Staged tuning: 25 trials each, then 75 for CatBoost and LightGBM, then stop | Spend compute where the first stage says it may pay; stop when the curve is flat and the gain is below the noise band. |
| D22 | Freeze single CatBoost E066 | The best ensemble's advantage was a third of a standard error; ties go to the simpler model, by a rule written before the comparison was read. |

## 2. What was tried and rejected

| Idea | Evidence | Why it matters |
|---|---|---|
| Drop the two (or four) large-area outliers | E049–E054: +0.004 to +0.009 worse for linear models, +0.001 to +0.005 for XGBoost | The usual gain in public notebooks comes from also deleting them from validation. |
| KNN or global-median LotFrontage | E043–E048: differences at most 0.002 | Imputation method for one column with 18 % missing values barely matters here. |
| Ordinal quality codes as a stand-alone improvement | ablations: within ±0.001 | One-hot already carries the information. |
| RobustScaler versus StandardScaler | linear grid: at most 0.0001 | Irrelevant once skewed columns are log-transformed. |
| Ridge as a finalist | 0.005 behind Lasso in every fold | With about 300 one-hot columns, sparsity beats shrinkage. |
| Native categorical handling in CatBoost | worse in all five paired trials by 0.0016–0.0028, at 15–24 times the cost; study stopped at 5 of 25 trials | A partial result, reported as partial. |
| More XGBoost tuning | Stage 1 flat from trial 4, 0.0036 behind | The Stage 2 rule excluded it. |
| NNLS blend weights, stacking with a Ridge meta-learner | never better than the equal-weight average on either set of folds | Learned weights add variance; with two members and correlated errors there is little to learn. |
| Ensembles of six bases or of three boosters | slightly worse than CatBoost alone | Members of the same family have residual correlations of 0.94–0.997; they add cost, not diversity. |
| Two-sided year clipping | would have changed two valid test rows | A "fix" must be checked against what else it touches. |

## 3. Limitations to state yourself

1. **The model choice at the top is within noise.** Single CatBoost and the Lasso + CatBoost average differ by 0.001 with a standard error of about 0.003. Another analyst could defensibly ship the average.
2. **Expected error is about 0.12, not the holdout's 0.106.** The holdout has 219 rows and none of the four largest houses.
3. **Selection optimism was measured, not removed.** The fresh-fold check corrects the reported number, but the same development rows were used throughout; only the holdout is truly untouched, and it is small.
4. **The search space had edges.** Both Stage 2 bests sit at the 2,000-tree upper bound; a wider search might find slightly better settings. The expected gain is below the noise band.
5. **A few houses dominate the error.** Large, atypically cheap or expensive houses are predicted badly, and the test set contains at least one (Id 2550). CV cannot tell which way that prediction errs.
6. **Regression toward the mean.** Cheap houses are over-predicted and expensive ones under-predicted (part 5).
7. **One city, 2006–2010.** Nothing here says how the model would behave on other markets or years; time was treated as a feature, not as a split.
8. **Tooling.** Registry without locking, pickled model artefacts, Optuna study not exactly re-creatable.
9. **No leaderboard score.** By choice, the file was not submitted, so there is no external confirmation.

## 4. Questions you should expect, with short answers

**What is your validation scheme and why should I trust the number?** Five-fold CV on development rows with all fitted preprocessing inside the fold; finalists re-scored on 15 new folds; one frozen model scored once on a locked holdout; every score reproducible by a command; an independent auditor's findings and fixes in the repository.

**What is data leakage and where could it have happened here?** Any path by which information about validation or test targets, or statistics computed with validation rows, reaches training. Candidates here: imputing with statistics from all rows, choosing skewed columns on all rows, removing outliers from validation folds, fitting ensemble weights on the predictions they are then scored on, tuning on the holdout. Each has a structural guard (part 2a, section 4).

**Why did you not use the ensemble? Everyone on Kaggle does.** On a leaderboard a gain of 0.001 moves you up many places and costs nothing. As an estimate of real performance it was not distinguishable from zero in my data, and I committed to a rule before looking. I report both numbers.

**Why CatBoost over XGBoost and LightGBM?** On this data it was better at every stage (0.1181 against 0.1217 and 0.1231 on the selection folds; 0.0036 better than LightGBM on fresh folds, about two standard errors). Its trial scores were also much less sensitive to hyper-parameters. Symmetric trees are a strong regulariser on small data. That is an empirical statement about 1,241 rows, not a general ranking.

**Why is your CV score worse than the 0.11 I see in public notebooks?** Mostly protocol: many public scores remove outliers from all rows, fit preprocessing on train and test together, and report the best of many attempts on the same folds or on the public leaderboard. On my selection folds the best procedure scored 0.117; measured honestly it is 0.121.

**What would you do with another week?** Target the errors rather than the average: a model or features for the large-house and low-price segments, a monotonic constraint on area, quantile or interval predictions; a wider search on the tree count; a time-aware validation split to check stability across sale years.

**What did the audit actually catch?** Real defects in code that passed casual review: out-of-fold predictions overwritten under repeated CV (F1), NaN predictions scored silently (F2), the Id column dropped before a filter needed it (F9), finalist folds identical to selection folds (F20), `train --rows all` possible before the holdout was scored (F24), and a standard error that treated correlated folds as independent (F30).

## 5. A mental model to keep

1. Split off a test set first and lock it.
2. Anything that learns from rows goes inside the fold; only row-wise arithmetic stays outside.
3. A score you used for choosing is biased; measure the chosen thing again on data that did not choose it.
4. Compare models with paired differences and an honest standard error; decide the rule before reading the result.
5. Prefer the simpler model when the data cannot tell them apart, and say that this is what you did.
6. Report what failed and what you cannot know.
