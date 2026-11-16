# Study guide, part 4: from finalists to a frozen model

Parts 2 and 3 produced candidates. This part explains how one of them became the final model, why the choice was made the way it was, what the holdout did and did not tell us, and how the submission file was produced. The records are ruling D22 in `docs/EXPERIMENT_HISTORY.md`, section 9 and 10 of `docs/VALIDATION_AUDIT.md`, registry ids E076 to E083, and `experiments/holdout_usage.log`.

## 1. Why the best score on the selection folds is not the answer

By the time the candidates existed, the five selection folds (seed 42) had been used to compare 136 linear grid points, 180 boosting trials and 12 ensembles. Each comparison picks the configuration that happens to suit those five particular splits. The picked score is therefore biased low: it contains the true quality of the configuration plus the luck that made it win. This is the winner's curse, and it grows with the number of things compared.

The cure is cheap: score the chosen configurations again on folds they were not chosen on. `experiments/finalists/run_finalists.py` rebuilds each candidate from the registry and runs `RepeatedKFold(n_splits=5, n_repeats=3, random_state=7)`. Seed 7 gives partitions that share no fold with the seed-42 partition (audit finding F20 was precisely that the first version reused seed 42, which made one of the three repeats identical to the selection folds).

The result, for the eight candidates:

| model | selection folds | fresh folds | optimism |
|---|---|---|---|
| average Lasso + CatBoost (E068) | 0.1173 | 0.1207 | +0.0033 |
| nested stack Lasso + CatBoost (E072) | 0.1175 | 0.1216 | +0.0042 |
| CatBoost, 75 trials (E066) | 0.1181 | 0.1217 | +0.0036 |
| NNLS blend Lasso + CatBoost (E069) | 0.1180 | 0.1224 | +0.0044 |
| CatBoost, 25 trials (E060) | 0.1195 | 0.1233 | +0.0038 |
| LightGBM, 75 trials (E067) | 0.1217 | 0.1252 | +0.0035 |
| Lasso (E056) | 0.1255 | 0.1282 | +0.0027 |

Every candidate got worse, by 0.0027 to 0.0044. Two things are mixed in that shift. One is the fold draw: the seed-7 partitions are simply a little harder for all models (the least-tuned model, Lasso, also moved by 0.0027). The other is the winner's curse, which should be larger for configurations picked from more alternatives (the blend and the stack, picked among ensembles built on bases that were themselves picked, moved most). You cannot separate the two exactly, and you do not need to: the fresh-fold number is the one to report.

## 2. Comparing two models properly: paired differences

Fold scores vary a lot (for CatBoost from 0.103 to 0.150 across the 15 fresh folds) because some folds contain hard houses. Comparing two means with that spread would hide any real difference. But both models are scored on the same folds, and a hard fold is hard for both. So the comparison is made on the per-fold difference d_i = score_A,i − score_B,i, which removes the fold difficulty.

For the average (E068) against single CatBoost (E066): mean difference −0.0010, standard deviation of the 15 differences 0.0068.

How precise is that mean? Three answers, from most to least optimistic:

- Naive: sd / sqrt(15) = 0.0018. This treats the 15 folds as independent measurements. They are not: the three repeats cut the same 1,241 houses three ways, and within a repeat the training sets overlap by 75 %.
- The auditor's choice (finding F30): sd / sqrt(5) = 0.0031, on the argument that the three repeats agree with each other almost perfectly and so contribute little new information beyond one 5-fold run.
- Nadeau and Bengio's corrected resampled variance: multiply the variance by (1/15 + n_test/n_train) = (1/15 + 248/993) = 0.316 instead of 1/15, giving 0.0068 × sqrt(0.316) = 0.0038.

Whichever you take, −0.0010 is between a quarter and a half of one standard error. The data cannot tell these two procedures apart. It also cannot exclude that the average is really better by 0.002 or 0.003; "not significant" is not "equal".

Contrast with CatBoost 75 trials against CatBoost 25 trials: difference 0.0016 with a standard deviation of only 0.0025, because the two models are almost the same model and their fold scores move together. That is 1.1 to 2.5 standard errors depending on the formula, and the three repeat means are 0.0017, 0.0015, 0.0016. Small, but consistent.

## 3. The decision rule was written before the data was seen

If you look at a table and then invent the rule that justifies your favourite row, the rule is worthless. The auditor therefore wrote the rule into `docs/VALIDATION_AUDIT.md` before reading the finalist records:

> Single CatBoost stays unless an ensemble is at least 0.0010 better on the fresh folds, and the difference exceeds one standard error (sd/sqrt(5)), and it wins at least 10 of the 15 folds with all three repeat means negative.

The reasoning behind the asymmetry (the simple model wins ties): an ensemble costs a second pipeline to maintain, explain and audit; its seed-42 advantage is inflated by having been the best of eight ensembles; and the project's standing rule since audit round 4 is that differences under 0.003 are not wins.

The average met two conditions (−0.0010; all three repeats negative) and failed two (one third of a standard error; 9 folds of 15). Ruling D22 froze single CatBoost E066.

Be precise when you explain this: it is a parsimony decision. The honest sentence is "the average had the best point estimate on both sets of folds, the difference was far inside the noise, so I kept the simpler model", not "the ensemble did not help".

A second argument, specific to this data, is about the test set. Test house Id 2550 has 5,095 square feet of living area, larger than all but one training house. A linear model multiplies area by a coefficient and so extrapolates; the two training houses of that kind (Ids 524 and 1299) sold far below what a linear trend predicts. A tree model predicts for Id 2550 what it learned for the largest houses it has seen. Neither is guaranteed to be right, but the tree's failure mode is bounded.

## 4. What "frozen" means

D22 records the registry id (E066), the configuration hash (`be58905cb40a`), the builder (`houseprices.models:build_catboost`), the six hyper-parameters at full precision, the feature set, the row filter and the target. The freeze was committed to Git (`23f56ac`) before anything else happened, so the history proves the order of events. From that commit on, nothing about the procedure may change in response to what the holdout says. If the holdout had come back terrible, the correct response would have been to look for a bug and report the bad number, not to retune.

## 5. The single holdout evaluation

Before the evaluation the auditor also wrote down what result would be consistent with the cross-validated estimate: centre 0.121, consistent from 0.100 to 0.145, a bad draw up to 0.150, and outside 0.095 to 0.150 a sign of a data or pipeline fault.

Then, in order:

1. `python -m houseprices.train --experiment E066` fitted the frozen pipeline on the 1,241 development rows and saved `outputs/models/E066_development.joblib` with a JSON sidecar (rows, hash, commit, library versions).
2. `experiments/finalists/score_holdout.py` checked the sidecar (right experiment, development rows only, no override), then called `validation.score_holdout_once` once.

`score_holdout_once` is the guard described in part 2a: it refuses a frame whose Ids are not exactly the 219 registered holdout Ids, refuses a second call, and appends the event to `experiments/holdout_usage.log`.

Result: **RMSE 0.10645** on 219 houses. Mean residual +0.004 (no bias), mean absolute residual 0.076, three houses with an absolute error above 0.3, median absolute error in dollars 5.3 %.

## 6. Why the holdout score is better than the CV estimate, and why that changes nothing

0.1065 against 0.1217 looks like good news. Treat it with the same suspicion you would apply to bad news.

- Sampling error. An RMSE over 219 houses is noisy because a handful of large errors dominate a squared-error average. The auditor's bootstrap of the 219 squared residuals gives a 95 % interval of 0.092 to 0.121 (standard deviation 0.007). The CV estimate of 0.1217 sits just above that interval, so noise alone is not the whole explanation; the next point is.
- The hard houses are elsewhere. The four houses above 4,000 square feet (Ids 524, 692, 1183, 1299) carry 8.5 % of the development-set out-of-fold squared error and raise its pooled RMSE by 0.005 (part 5), and all four are development rows; the largest house in the holdout has 3,627 square feet. The stratification of the holdout was by price decile, not by "difficulty". Of the 15 fresh folds, the five that contain none of them average 0.114 (0.103 to 0.125) and the ten that contain at least one average 0.126 (up to 0.150); the holdout, with none, sits in the first group.
- The model fitted for the holdout used 1,241 rows, CV models about 993. More data helps a little.

So the holdout says: the cross-validated estimate was not inflated by leakage or by overfitting the folds; nothing is broken. It does not say the model's expected error is 0.106. The number to quote for expected performance on new houses remains about 0.12, and you should expect Kaggle's test set, which contains at least one giant house, to behave more like the CV estimate than like the holdout.

The project had predicted (hypothesis H4) that the holdout would land within 0.01 of the fresh-fold mean. It landed 0.015 away. The prediction was wrong and the experiment history says so.

## 7. The final fit and the submission file

Once a holdout has been scored it has no further use as a test, and its 219 rows are worth more as training data. `train --rows all` refits the frozen configuration on all 1,460 rows. The code refuses this before the holdout has been scored (audit finding F24): a model that has seen the holdout must never be scored on it.

`python -m houseprices.predict --model outputs/models/E066_all.joblib` then applies the same cleaning and feature code to `test.csv`, predicts on the log scale, converts with `expm1`, checks that every price is finite and positive, aligns the rows to the Ids of `sample_submission.csv` and writes `outputs/submissions/E066_all.csv` (1,459 rows, prices from 41,901 to 524,842, median 156,611).

Things to notice:

- There is no cross-validated score for this last model and there cannot be one. Its expected error is inferred from the procedure's CV score. That is why the procedure, not a fitted object, is what gets frozen and evaluated.
- The predicted prices stop at 525 thousand while training prices reach 755 thousand. A model that minimises squared error on the log scale shrinks its predictions toward the centre; part 5 shows the same pattern in the residuals by price decile.
- The file was not submitted. Kaggle's public leaderboard is one more score on one more sample; using it to adjust the model would turn it into a tuning set.

## Self-check questions

1. Why did every finalist score worse on the seed-7 folds, including Lasso, which was chosen from a small grid?
2. The average beat CatBoost by 0.0010 on fresh folds and by 0.0007 on the selection folds. Why was it not chosen?
3. Why is sd/sqrt(15) too small a standard error for a 5x3 repeated cross-validation?
4. What would you have done if the holdout RMSE had been 0.16?
5. The holdout RMSE (0.1065) is lower than the CV estimate (0.1217). Which do you report as expected performance, and why?
6. Why is `train --rows all` refused before the holdout is scored?
7. Why is the decision rule's timing (written before the table was read) part of the evidence?

## Answers

1. Two effects: the seed-7 partitions are a slightly harder draw for all models, and scores chosen as the best of many on seed 42 include favourable luck that does not repeat. Lasso's shift (0.0027) is mostly the first effect.
2. The difference is a quarter to a half of its standard error; the pre-registered rule required more than one standard error and 10 of 15 folds; ties go to the simpler model. It is a parsimony choice and the report says the average may be slightly better.
3. The 15 fold scores share the same 1,241 rows; repeats re-cut the same data and training sets overlap heavily, so the scores are positively correlated and the effective number of independent measurements is much smaller than 15.
4. Look for a fault (wrong rows, a preprocessing difference between fit and predict, a data problem in the holdout), report the number as it is, and not retune. If anything were changed because of it, the holdout would no longer be an unbiased estimate and the report would have to say so.
5. The CV estimate, about 0.12. The holdout is a single sample of 219 houses that happens to contain none of the four hardest houses; it is a consistency check that came out on the favourable side of its expected range.
6. A model fitted on all rows has seen the holdout rows; scoring it there would measure memorisation. The guard makes the mistake impossible instead of merely discouraged.
7. A rule chosen after seeing the results can always be made to prefer any candidate. Committing the rule first (it is in the audit file, written before the finalist records were read) makes the choice a test, not a rationalisation.
