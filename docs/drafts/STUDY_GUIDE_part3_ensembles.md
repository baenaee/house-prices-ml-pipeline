# Study guide, part 3: ensembles, blending and nested stacking

Code: `src/houseprices/ensemble.py`, `experiments/ensemble/run_ensembles.py`, `tests/test_ensemble.py`.
Scores are RMSE on log1p(SalePrice), which is the Kaggle RMSLE. Lower is better.

## 1. Why ensembles help

A model's expected squared error splits into bias, variance and irreducible noise. Averaging several models does not change the noise and barely changes the shared bias. It reduces the variance part, and only the part that is not common to all members.

For two predictors with equal error variance and error correlation rho, the error variance of their average is sigma^2 (1 + rho) / 2. With rho = 1 there is no gain. With rho = 0 the variance halves. The whole game is therefore low error correlation at similar accuracy. A much worse model rarely helps, because it adds its own bias.

Our models differ in kind. Ridge, Lasso and ElasticNet are penalised linear models on one-hot and scaled inputs. XGBoost, LightGBM and CatBoost are boosted trees that capture thresholds and interactions. Within a family the errors are very similar. Across families they differ more.

### What the two correlations tell you

| Quantity | Computed on | Reads as |
|---|---|---|
| Prediction correlation | OOF predictions of two models | Almost always above 0.9 here, because every model predicts the same price scale. Nearly useless for judging diversity. |
| Residual correlation | prediction minus truth | The informative one. Near 1 means the models make the same mistakes. Lower values mean blending can cancel errors. |

In the dry-run on 150 rows, Lasso and ElasticNet had residual correlation about 0.97, and linear versus tree pairs were around 0.5 to 0.7. Those dry-run numbers are plumbing checks only. The real values come from the full run. The expected pattern is the point: pairing across families beats pairing within a family.

The diversity report uses single-pass OOF predictions, one per row, from the same folds as the registry. It is a diagnostic. Nothing is fitted on it and then scored on the same rows.

## 2. Averaging versus learned blend weights

An equal-weight average has no parameters, so it cannot overfit. It is the naive reference. A learned blend picks weights w that minimise the squared error of the weighted prediction against the target.

Constraints matter:

- **Non-negative:** without it, least squares may give a model a negative weight and "subtract" it. That usually exploits noise in the training predictions, and it does not generalise.
- **Sum to one:** keeps the blend on the scale of the target and removes a free scaling parameter. The code solves it with a constrained optimiser (SLSQP). With only non-negativity it uses NNLS, non-negative least squares.

The weights are learned inside `fit` from inner out-of-fold predictions of the rows received. The outer validation fold never influences them.

### Audit F18: weights chosen on the folds you score

If someone picks fixed weights by looking at OOF scores from earlier experiments on the same development folds, the blend's CV score is optimistic. The weights have seen the validation targets indirectly. We therefore report only `weights=None` (learned in fit) or equal weights, which are fixed before looking at any score.

## 3. Stacking with a Ridge meta-learner

Stacking replaces the weighted average with a second model. It takes the base predictions as features and learns the target. We use Ridge with alpha 1 and `passthrough=False`.

- **Ridge:** a low-capacity meta-learner. With six highly correlated columns, an unregularised or flexible meta-model would chase noise. Ridge shrinks and stabilises the coefficients.
- **passthrough=False:** the meta-learner sees only base predictions, not the raw features. Passing raw features back in would give it room to relearn the problem and overfit. It also would not work on our mixed-type frames. This was a director decision, and the code defaults to it.

Unlike a blend, the stack allows weights that do not sum to one and may be negative, plus an intercept. That makes it more flexible and riskier.

## 4. Binding correction 1: nested stacking, one outer fold step by step

The user's correction: the meta-learner must never be scored on rows it was trained on. The classic mistake is to fit bases, collect OOF predictions for the whole development set, fit the meta-learner on them, and report CV on the same rows. The meta-learner has already seen those rows' targets.

Our fix is nested CV. The inner loop lives inside `fit`, so the outer protocol (`validation.cross_validate`) stays the single source of truth, and any ensemble reproduces through `python -m houseprices.evaluate --experiment <id>`.

Worked example, rounded. Development set 1,241 rows. Outer fold 3 of 5 holds out about 248 rows. The outer-train part has about 993 rows. A training-row filter, if any, has already removed outlier rows from these 993 and never touches the 248.

```
DEVELOPMENT (1241 rows)
|-- OUTER VALIDATION, fold 3 (248 rows)  ----- hidden until the very end
|-- OUTER TRAIN (993 rows) -> stack.fit(X_tr, y_tr)
      Step 1  inner 5-fold on the 993 rows (about 199 rows per inner fold)
              for each inner fold i and each base b:
                  fit b on the other 4 inner folds (about 794 rows)
                  predict inner fold i      -> column b of the OOF matrix
              result: a 993 x k matrix; every entry came from a model
              that did not see that row
      Step 2  fit the meta-learner (or blend weights) on that matrix
              with target y_tr
      Step 3  refit every base on all 993 rows
stack.predict(outer validation 248 rows):
      bases (Step 3) predict -> meta-learner (Step 2) combines -> score RMSE
```

The 248 outer validation rows are used exactly once, for scoring. Nothing the meta-learner learned involved them, so the fold score is an honest estimate. The meta-learner in Step 2 learned on out-of-sample base predictions, which look like the predictions the refit bases will make on unseen rows. That match in distribution is the reason to use OOF rather than in-sample predictions: in-sample predictions are too good, so the meta-learner would over-trust the flexible model.

The data flow in the other direction matters too. The meta-learner is trained on Step 1 predictions made by models trained on 80% of the outer-train rows, but applied to predictions from models trained on 100% of them. This small mismatch is standard and harmless.

## 5. Why the "OOF-stack" convenience score is optimistic

`optimistic_oof_stack_score` computes one OOF matrix on the full development set, fits the meta-learner on all of it, and scores the meta-learner on those same rows. It is quick, and it is a plausible-looking number. But the meta-learner was trained on the targets it is scored against. With a few Ridge coefficients the optimism is small. With a flexible meta-model or many bases it can be large. `in_sample_stack_score` is worse: the bases themselves are also scored on their own training rows.

Audit finding F17 asked for a guard against registering these numbers. We did two things in `ensemble.py`:

1. Both functions return an `OptimisticScore`, a float subclass with the attribute `registerable = False`, a `kind` label, and a repr that says "NEVER REGISTER".
2. It has no `fold_scores` or `mean`, so `_common.run_and_register(result=...)` fails with an `AttributeError` before anything is written. It is also always truthy. Otherwise a score of exactly 0.0 would be falsy and make the helper silently run a real CV instead.

The limit of the guard: someone could still build a registry `Experiment` by hand with that float as `cv_mean`. The registry has no check for it, which is why the auditor marked F17 as low. The runner never calls these functions.

## 6. The memoriser leak test and its blind spot (audit F16)

To prove the nested design catches the leak, a test uses a "memoriser" base. It returns the exact target for rows it was trained on and the training mean for any unseen row. The in-sample stack score of a memoriser is near zero (perfect cheating). The nested score must be about as bad as predicting the mean.

The blind spot (F16): with a memoriser as the only base, a leaky stack scores the same as the honest one. For unseen rows the memoriser returns a constant, so whatever weight the meta-learner gave it is irrelevant. The test cannot tell whether the meta-learner was trained on in-sample predictions. Another test (inner OOF rows are never seen by the model that predicts them) does protect the OOF generation, but the meta-training-data leak had no direct test in our file.

Fix: use two bases, a memoriser and a Ridge. In the leaky version the meta-learner sees the memoriser's perfect in-sample predictions, gives it nearly all the weight, and then collapses on unseen rows, because the memoriser returns a constant there. Ridge is the useful base that the honest stack should lean on. Expected pattern per the auditor's probe:

| Variant | Score relative to Ridge alone |
|---|---|
| Honest nested stack | within about 15% |
| Leaky mutant | more than 3 times worse |

The point of the second base is that the leak now has something to damage.

## 7. Nested cost arithmetic

Per outer fold, with k bases and 5 inner folds: 5 x k inner fits plus k refits, so 6k fits. Over 5 outer folds that is 30k fits. A plain 5-fold CV of the same bases costs 5k fits.

| Bases | Plain CV fits | Nested fits per outer fold | Nested fits total |
|---|---|---|---|
| 2 | 10 | 12 | 60 |
| 6 | 30 | 36 | 180 |

Inner fits use about 64% of the rows (80% of 80%). Measured against one fit at 80%, the nested cost is about (5 x 0.64 + 0.8) / 0.8 = 5 times a single 5-fold CV. The runner uses 5.5 as a slightly conservative factor, because fit time is not perfectly linear in rows. Estimate of an ensemble = 5.5 x the sum of its bases' registered CV times. CatBoost dominates the total. The runner prints this estimate, and takes `--budget-minutes` (default 40), skipping experiments that would overrun it.

## 8. How the runner compares an ensemble to its best member

For each ensemble the runner takes the best single member among its own bases (lowest registered cv_mean), and also the best base overall. Both were scored on the same five folds as the ensemble, because the seed and splitter are fixed. So the comparison is paired:

1. d_i = ensemble score minus member score on fold i, for i = 1..5.
2. Mean difference = average of d_i. Negative means the ensemble is better.
3. Paired standard error = std(d_i, ddof 1) / sqrt(5).
4. One-SE rule: mean difference below minus one SE means retain. Above plus one SE means reject. Otherwise investigate.

Why paired: fold difficulty varies a lot (fold scores differ by a few hundredths) and mostly cancels in the differences. Comparing the two cv_means with their unpaired standard deviations would hide real gains. Why only one SE and five folds: the SE from five differences is itself noisy, so a retain means "worth confirming" and not "proven". The director re-tests finalists with RepeatedKFold 5x3.

Decisions are also made against the best member, not against the average of members. An ensemble that beats the average member but not the best one is not worth its extra fitting time.

## 9. Self-check questions

1. For two models with equal error variance, how does the variance of their average depend on error correlation? *Variance falls to sigma^2 (1 + rho) / 2. At rho = 1 there is no gain, at rho = 0 it halves.*
2. Why is prediction correlation a poor diversity measure here? *All models predict the same price scale, so it is above 0.9 for almost any pair. Residual correlation removes the shared signal and shows the shared mistakes.*
3. Why constrain blend weights to be non-negative and sum to one? *Negative weights exploit noise, and a free sum adds a scale parameter. The constraints act as regularisation.*
4. What is the F18 risk? *Choosing weights from OOF scores on the same folds later used to report the blend makes the score optimistic. We learn weights inside `fit` or use equal weights.*
5. Why use Ridge and `passthrough=False` in the meta-learner? *Base predictions are strongly correlated, so a low-capacity regularised model is stable, and without raw features the meta-model cannot relearn the whole problem.*
6. Walk through one outer fold of nested stacking. *Inner K-fold on outer-train rows gives an OOF matrix. Fit the meta-learner on it. Refit the bases on all outer-train rows. Predict the outer validation fold once through bases then meta.*
7. Why must the meta-learner train on OOF and not in-sample base predictions? *In-sample predictions are too accurate, so it over-trusts flexible models, then fails on unseen rows where bases are less accurate.*
8. What does `OptimisticScore` prevent, and what does it not? *It stops the optimistic number from flowing through `run_and_register`, and a 0.0 value from triggering a real CV. It cannot stop someone building a registry record by hand.*
9. What was the F16 blind spot and how is it closed? *A lone memoriser returns a constant on unseen rows, so a leaky and an honest stack score alike. Adding a Ridge base gives the leak something to damage: honest stays near Ridge, leaky is much worse.*
10. How many fits does a nested stack of six bases need per outer fold and in total? *6 x 6 = 36 per outer fold and 180 over five folds, versus 30 for plain CV.*
