<!-- STATUS: updated 2026-11-17 with final bases (sections 9, 10) and the freeze discussion (section 11; director ruling placeholder pending). Sections 1 and 8b keep the Stage 1 story. -->

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

Stage 1 values (superseded by the final-bases table in section 9.1), from `experiments/ensemble/diversity_20261113_121446.csv`: single-pass OOF predictions of the six tuned Stage 1 bases (E055 to E060) on the working feature set, seed-42 folds. Each base's recomputed cv_mean matches its registered value to six decimals.

Residual correlation (3 decimals):

| | ridge | lasso | enet | xgb | lgbm | cat |
|---|---|---|---|---|---|---|
| ridge | 1.000 | 0.970 | 0.975 | 0.825 | 0.823 | 0.852 |
| lasso | 0.970 | 1.000 | 0.997 | 0.831 | 0.829 | 0.860 |
| elasticnet | 0.975 | 0.997 | 1.000 | 0.833 | 0.834 | 0.862 |
| xgboost | 0.825 | 0.831 | 0.833 | 1.000 | 0.962 | 0.946 |
| lightgbm | 0.823 | 0.829 | 0.834 | 0.962 | 1.000 | 0.946 |
| catboost | 0.852 | 0.860 | 0.862 | 0.946 | 0.946 | 1.000 |

Prediction correlation (3 decimals):

| | ridge | lasso | enet | xgb | lgbm | cat |
|---|---|---|---|---|---|---|
| ridge | 1.000 | 0.996 | 0.997 | 0.980 | 0.980 | 0.983 |
| lasso | 0.996 | 1.000 | 1.000 | 0.982 | 0.981 | 0.985 |
| elasticnet | 0.997 | 1.000 | 1.000 | 0.982 | 0.982 | 0.985 |
| xgboost | 0.980 | 0.982 | 0.982 | 1.000 | 0.996 | 0.994 |
| lightgbm | 0.980 | 0.981 | 0.982 | 0.996 | 1.000 | 0.994 |
| catboost | 0.983 | 0.985 | 0.985 | 0.994 | 0.994 | 1.000 |

Per-base RMSE: ridge 0.1319, lasso 0.1272, elasticnet 0.1272, xgboost 0.1242, lightgbm 0.1256, catboost 0.1204.

Reading:

- Prediction correlation is between 0.980 and 1.000 for every pair, so it says nothing about diversity, as predicted above.
- Within a family the models are near duplicates: Lasso and ElasticNet at 0.997, XGBoost and LightGBM at 0.962, CatBoost with either at 0.946.
- Cross-family pairs are the most complementary, with residual correlation 0.82 to 0.86. The lowest are Ridge with LightGBM (0.823) and Ridge with XGBoost (0.825).
- CatBoost is the most accurate base. Its best partners by diversity are the linear models (0.852 to 0.862), which is why Lasso plus CatBoost is the natural pair.
- Residual correlation is high everywhere, so expect modest ensemble gains.

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

## 8b. Worked example: average versus learned weights versus six-base blend (E061 to E064)

Stage 1 history. The final-bases rerun (E068 to E075, section 9) and the fresh-fold check (section 10) supersede these numbers.

These four records were registered ahead of the planned full ensemble run, so they are labelled premature in the experiment history. They are valid CV runs. Caveat: they use the Stage 1 tuned bases (E055 to E060), the seed-42 five-fold split and a single repeat. With five folds, a paired SE is noisy. Treat the pattern as a hypothesis for the RepeatedKFold 5x3 test, not as a conclusion.

Reference: best single base is CatBoost E060 at 0.11946. Differences are ensemble minus CatBoost, averaged over the five folds, with the paired standard error.

| Id | Ensemble | cv_mean | Difference vs CatBoost | Paired SE | Runner decision |
|---|---|---|---|---|---|
| E061 | Equal-weight average, Lasso + CatBoost | 0.11807 | -0.0014 | 0.0019 | investigate |
| E062 | Learned blend, Lasso + CatBoost | 0.11902 | -0.0004 | 0.0016 | investigate |
| E063 | Learned blend, all six bases | 0.11943 | -0.00003 | 0.0019 | investigate |
| E064 | Learned blend, Ridge + Lasso + ElasticNet | 0.12585 | n/a | n/a | reject |

E064 is compared with its own best member, Lasso (0.12545), not with CatBoost, and is worse by more than one SE.

What to take from it:

- **Average versus learned weights.** The naive average scored slightly better than the learned blend of the same two models (0.11807 against 0.11902). Their difference is 0.0009 with an SE of 0.0007, so it is within noise. With two highly accurate partners, estimating weights from about 1,000 inner OOF rows adds a little variance that equal weights avoid. This is the usual reason to keep the naive reference.
- **Six bases did not beat two.** Adding Ridge, ElasticNet, XGBoost and LightGBM brought nothing over CatBoost alone. This agrees with the diversity tables: the extra models are near duplicates of Lasso or of the boosted trees.
- **Same-family blends do not help.** The linear-only blend is worse than its own best member, since the three linear models share almost all their errors (0.970 to 0.997).
- **One-SE rule in action.** No ensemble beats CatBoost by more than one SE, so none is "retain". All are "investigate" except the rejected one.

## 9. Final bases: ensembles after Stage 2 (E068 to E075)

Stage 2 tuning replaced CatBoost E060 (0.11946) with E066 (0.11807) and LightGBM E059 (0.12442) with E067 (0.12173). The ensembles were rerun on the six final bases (E055 Ridge, E056 Lasso, E057 ElasticNet, E058 XGBoost, E067 LightGBM, E066 CatBoost), same seed-42 five-fold split, nested protocol as in section 4. Sections 1 and 8b above are kept as the Stage 1 story. Where a number there is superseded, this section gives the new one.

### 9.1 Diversity with the final bases

Source: `experiments/ensemble/diversity_20261116_234858.csv`. Pooled OOF RMSE (one prediction per row, so slightly above the fold-mean values in the registry) and residual correlation, 3 decimals.

| | ridge | lasso | enet | xgb | lgbm | cat |
|---|---|---|---|---|---|---|
| ridge | 1.000 | 0.970 | 0.975 | 0.825 | 0.827 | 0.846 |
| lasso | 0.970 | 1.000 | 0.997 | 0.831 | 0.837 | 0.856 |
| elasticnet | 0.975 | 0.997 | 1.000 | 0.833 | 0.840 | 0.860 |
| xgboost | 0.825 | 0.831 | 0.833 | 1.000 | 0.972 | 0.939 |
| lightgbm | 0.827 | 0.837 | 0.840 | 0.972 | 1.000 | 0.951 |
| catboost | 0.846 | 0.856 | 0.860 | 0.939 | 0.951 | 1.000 |

| Base | Pooled OOF RMSE | Mean residual corr. with the other five |
|---|---|---|
| ridge | 0.1319 | 0.888 |
| lasso | 0.1272 | 0.898 |
| elasticnet | 0.1272 | 0.901 |
| xgboost | 0.1242 | 0.880 |
| lightgbm | 0.1228 | 0.885 |
| catboost | 0.1190 | 0.890 |

Prediction correlation is again between 0.980 and 1.000 for every pair. The structure is unchanged: linear to linear 0.970 to 0.997, tree to tree 0.939 to 0.972, linear to tree 0.825 to 0.860. Against the Stage 1 table the tuned trees are more accurate (CatBoost 0.1204 to 0.1190, LightGBM 0.1256 to 0.1228) and the CatBoost-Lasso residual correlation moved from 0.860 to 0.856. Diversity barely changed, but CatBoost got better, and that is what matters below.

### 9.2 Results

Paired difference is ensemble minus its best member on the same five folds (negative is better), with the paired SE. For every ensemble except E074 the best member is CatBoost E066 (0.118072). For E074, three linear models, it is Lasso E056 (0.12545).

| Id | Ensemble | CV mean | Diff vs best member | Paired SE | Diff / SE | Verdict |
|---|---|---|---|---|---|---|
| E068 | average, Lasso + CatBoost | 0.117344 | -0.00073 | 0.00194 | -0.37 | investigate |
| E072 | stack (Ridge meta), Lasso + CatBoost | 0.117479 | -0.00059 | 0.00165 | -0.36 | investigate |
| E069 | NNLS blend, Lasso + CatBoost | 0.118022 | -0.00005 | 0.00174 | -0.03 | investigate |
| E073 | stack, six bases | 0.118464 | +0.00039 | 0.00222 | +0.18 | investigate |
| E075 | stack, three boosters | 0.118582 | +0.00051 | 0.00124 | +0.41 | investigate |
| E071 | NNLS blend, three boosters | 0.118722 | +0.00065 | 0.00094 | +0.69 | investigate |
| E070 | NNLS blend, six bases | 0.118937 | +0.00087 | 0.00226 | +0.38 | investigate |
| E074 | stack, three linear models | 0.125927 | +0.00048 | 0.00035 | +1.37 | reject |

Nothing reaches "retain" (a difference below minus one SE). The best, E068, is at minus 0.37 SE. E074 is worse than its own best member by more than one SE, so it is rejected, as E064 was in Stage 1.

### 9.3 Reading

- **The gain from Lasso shrank as CatBoost improved.** In Stage 1 the equal-weight Lasso + CatBoost average was 0.11807 against CatBoost 0.11946 (E061, difference -0.0014). With the Stage 2 CatBoost it is 0.11734 against 0.11807 (E068, difference -0.0007). The ensemble improved by 0.0007 while its best member improved by 0.0014, so the margin halved.
- **Six bases and three boosters are slightly worse than CatBoost alone.** E070, E071, E073 and E075 all have positive differences (+0.0004 to +0.0009), each within one SE. Ridge, ElasticNet, XGBoost and LightGBM are near duplicates of Lasso or CatBoost (residual correlation up to 0.997 within the linear family and 0.951 with CatBoost), so they add weight-estimation variance and almost no diversity.
- **Learned weights did not beat equal weights.** For the Lasso + CatBoost pair the order is average (0.11734), stack (0.11748), NNLS blend (0.11802). With two accurate partners and about 1,000 inner OOF rows, estimating weights costs more than it gains.

### 9.4 Why: the two-model variance formula, with numbers

For unequal members, let Lasso and CatBoost have error standard deviations s1 and s2 and residual correlation rho. With weight w on CatBoost the error variance of the combination is

```
Var(w) = w^2 s2^2 + (1-w)^2 s1^2 + 2 w (1-w) rho s1 s2
```

At w = 1/2 and s1 = s2 this reduces to the formula of section 1, sigma^2 (1 + rho) / 2. The minimising weight is

```
w* = (s1^2 - rho s1 s2) / (s1^2 + s2^2 - 2 rho s1 s2)
```

Inputs from `diversity_20261116_234858.csv`: s1 = 0.127183 (Lasso), s2 = 0.119007 (CatBoost), rho = 0.85637.

| Quantity | Value |
|---|---|
| Predicted RMSE of the equal-weight average | 0.11860 (pooled), which is 0.00041 below CatBoost pooled 0.11901 |
| Optimal weight | 0.728 on CatBoost, 0.272 on Lasso |
| Predicted RMSE at the optimal weight | 0.11763 (pooled), which is 0.00138 below CatBoost pooled |
| Break-even correlation (the equal-weight average ties CatBoost) | rho = (3 s2^2 - s1^2) / (2 s1 s2) = 0.869 |
| Measured, E068 average (fold mean) | 0.117344, which is 0.00073 below E066 0.118072 |
| Measured, E069 NNLS blend (fold mean) | 0.118022, which is 0.00005 below E066 |

Reading it honestly:

- The inputs are pooled OOF RMSEs and the measurements are means of five fold RMSEs. They are not the same quantity: the fold mean is lower (CatBoost 0.118072 against 0.119007, Lasso 0.12545 against 0.12718). So compare differences, not levels. The predicted gain of the average is -0.0004 and the measured gain is -0.0007: same sign and order of magnitude. The formula assumes a fixed correlation and ignores the weights being estimated, so a 0.0003 gap is not a contradiction.
- The average is close to break-even. The observed rho (0.856) is only 0.013 below 0.869, so the average beats CatBoost by a thin margin that would vanish at 0.869. This is the mechanism behind the shrinkage: the weaker Lasso (0.127 against 0.119) must pay for itself through decorrelation, and the better CatBoost gets, the less it can. With the Stage 1 inputs (CatBoost 0.1204, Lasso 0.1272, rho 0.860 from section 1, all rounded) the same formula predicts a gain of about -0.0010, against about -0.0004 now. The measured gains went from -0.0014 to -0.0007. The formula gets the direction and roughly the size of the shrinkage.
- The optimal weight is 0.73 on CatBoost, not 0.5, and would give a larger predicted gain (-0.0014 pooled). But that weight is computed from the same residuals it would be scored on, and a learned blend has to estimate it from inner folds. Measured, the NNLS blend (E069) did worse than the equal-weight average, by 0.0007. Averaging at 0.5 forgoes about 0.001 of the theoretical optimum and has no parameter to overfit.
- The same structure holds for more members: each new member must be accurate and decorrelated from the existing average. Boosters correlate 0.94 to 0.97 with CatBoost and the other linear models 0.97 to 0.997 with Lasso, so additions bring mostly shared errors plus their own bias.

## 10. Finalists on fresh folds (E076 to E083)

Rule (decision D17, audit F20): finalists are re-scored on folds disjoint from the selection partition. Selection used seed-42 five-fold. Finalists use RepeatedKFold 5 splits x 3 repeats, seed 7, giving 15 fold scores each. Each finalist has the same configuration as its source experiment, and only the splitter changes. Sources: `experiments/finalists/finalists_20261117_003230.csv` and registry records E076 to E083 (the `notes` field names the source).

### 10.1 Selection score, fresh score, optimism

Optimism is fresh score minus selection score (positive means the selection-fold number was too good).

| Finalist | Source | Model | Selection (seed 42) | Fresh (seed 7, 15 folds) | Fold std | Optimism |
|---|---|---|---|---|---|---|
| E077 | E068 | average Lasso + CatBoost | 0.11734 | 0.12066 | 0.01433 | 0.00331 |
| E081 | E061 | average Lasso + Stage 1 CatBoost | 0.11807 | 0.12147 | 0.01488 | 0.00340 |
| E078 | E072 | stack Lasso + CatBoost | 0.11748 | 0.12164 | 0.01398 | 0.00416 |
| E076 | E066 | CatBoost Stage 2 (75 trials) | 0.11807 | 0.12166 | 0.01305 | 0.00359 |
| E079 | E069 | NNLS blend Lasso + CatBoost | 0.11802 | 0.12240 | 0.01393 | 0.00438 |
| E080 | E060 | CatBoost Stage 1 (25 trials) | 0.11946 | 0.12325 | 0.01286 | 0.00379 |
| E083 | E067 | LightGBM Stage 2 | 0.12173 | 0.12524 | 0.01320 | 0.00350 |
| E082 | E056 | Lasso grid | 0.12545 | 0.12816 | 0.01865 | 0.00271 |

Every finalist scores worse on fresh folds. The mean optimism is 0.0036 (range 0.0027 to 0.0044). This is the size of the selection effect: a configuration chosen as the best of many tries on the same folds is biased low. It exceeds the project's 0.003 rule for calling one model better than another, which is why finalists are never compared on selection scores. The two learned-weight finalists (stack E078 0.0042, blend E079 0.0044) have the largest optimism, which fits a meta-learner adding a little fit to the selection folds, but with differences of 0.001 this is a suggestion, not a finding.

### 10.2 Paired differences against single CatBoost E066

Fold i of every finalist is paired with fold i of E076, since all eight used the same seed-7 splitter. Difference is finalist minus E076 (negative is better). SE is the sd of the 15 differences divided by sqrt(15). "Folds won" counts folds where the finalist beat E076. The repeat columns are the mean difference over the five folds of each repeat.

| Finalist | Model | Mean diff | SD of diffs | SE (naive) | Folds won | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---|---|---|---|---|---|---|---|
| E077 | average E068 | -0.00101 | 0.00683 | 0.00176 | 9/15 | -0.00074 | -0.00102 | -0.00126 |
| E081 | average E061 (Stage 1 CatBoost) | -0.00020 | 0.00815 | 0.00210 | 9/15 | +0.00010 | -0.00018 | -0.00051 |
| E078 | stack pair E072 | -0.00002 | 0.00621 | 0.00160 | 10/15 | -0.00016 | +0.00060 | -0.00050 |
| E079 | blend pair E069 | +0.00074 | 0.00584 | 0.00151 | 9/15 | +0.00070 | +0.00111 | +0.00041 |
| E080 | CatBoost Stage 1 E060 | +0.00158 | 0.00250 | 0.00064 | 5/15 | +0.00165 | +0.00154 | +0.00157 |
| E083 | LightGBM E067 | +0.00357 | 0.00363 | 0.00094 | 4/15 | +0.00444 | +0.00275 | +0.00353 |
| E082 | Lasso E056 | +0.00650 | 0.01473 | 0.00380 | 6/15 | +0.00673 | +0.00670 | +0.00607 |

Reading:

- **The average E077 beats E066 in all three repeats** (-0.0007, -0.0010, -0.0013) and in 9 of 15 folds. The mean difference is -0.0010 with a naive SE of 0.0018, 0.57 SE from zero. It is not distinguishable from zero.
- **Stage 2 CatBoost beats Stage 1 CatBoost on fresh folds.** E080 minus E076 is +0.0016, SE 0.0006 (2.5 naive SEs), and positive in all three repeats (+0.0015 to +0.0017). E080 wins only 5 of 15 folds, so the gain is small per fold and consistent. This supports D21 on independent folds.
- **The stack pair and the Stage 1 average are tied with E066** (-0.00002 and -0.0002). The learned blend E079 is slightly worse (+0.0007).
- **LightGBM and Lasso are clearly worse than CatBoost** (+0.0036 and +0.0065).
- The finalists CSV column `paired_diff_vs_best` is measured against E077 (the best fresh score), not against E066, so its values differ from this table (for example +0.00101 for E076, which is E076 minus E077). Use the table here for comparisons against E066.

### 10.3 The 15 fold scores are not independent

The SE above treats the 15 differences as independent, and they are not. Each repeat splits the same 1,241 rows into five folds, so every row is in a test fold once per repeat, and training sets (80% of the rows each) overlap heavily across folds and repeats. Three repeats carry far less than three times the information of one 5-fold CV. The naive SE, sd/sqrt(15), understates the true standard error.

Nadeau and Bengio (2003) correct for this. For J = k x r = 15 resampled differences with sample variance s^2, the corrected variance of the mean difference is

```
Var_corrected = (1/J + n_test/n_train) * s^2
```

The second term does not shrink with more repeats. It stands for the dependence between folds, so more repeats can only reduce the first term.

For E077 against E076: s = 0.00683, J = 15, n_test = 248, n_train = 993 (5-fold on 1,241 rows). The factor is 1/15 + 248/993 = 0.0667 + 0.2497 = 0.3164, so

```
corrected SE = 0.00683 * sqrt(0.3164) = 0.00384
```

This is 2.2 times the naive 0.00176. The mean difference of -0.00101 is then 0.26 corrected SEs from zero (0.57 naive). The direction is the same and the evidence is weaker: the average-versus-CatBoost difference is far inside the noise of this protocol. The same correction on E080 minus E076 (s = 0.00250) gives a corrected SE of 0.00141 and a ratio of 1.1, so even the Stage 2 CatBoost gain is not conclusive on its own. The correction is an approximation derived for random resamples, so read the corrected SE as the more cautious of the two, not as exact.

## 11. What should be frozen? A balanced view

Candidates: (A) CatBoost E066 alone, (B) the equal-weight average of Lasso E056 and CatBoost E066 (the E068 configuration). The learned blend and the stack are not competitive: they cost far more fitting time on the fresh folds (E072 489.6 s and E069 529.0 s, against 97.4 s for the average and 84.3 s for CatBoost) and were not better.

For single CatBoost:

- It is the simplest candidate: one model, one set of hyper-parameters, one fit.
- No ensemble beats it by more than one SE on the selection folds, and none reaches the project's 0.003 rule on fresh folds (best, E077, -0.0010). With the fold-dependence correction the best difference is 0.26 SE.
- One model is easier to explain and audit (importances, partial dependence, SHAP).
- Trees do not extrapolate. On test-only extreme houses such as Id 2550, whose features lie outside the training range, a tree prediction stays within the range of the training targets, while a linear member can follow a slope beyond it. An equal-weight average passes half of the linear member's extrapolation through.
- The measured advantage of the average (0.0010) is inside the noise, so adding a second model would be paid for by complexity without evidence of a return.

For the equal-weight average:

- It was never worse than CatBoost in any of the three repeats (-0.0007, -0.0010, -0.0013) and won 9 of 15 folds.
- It has no learned weights, so there is no F18 risk and nothing to overfit. Its fit time is 97.4 s against 84.3 s.
- It hedges model-class risk. If the test distribution differs from the development one, two model classes with residual correlation 0.856 are less likely to fail together than one.
- By the variance formula in 9.4 the expected gain is positive as long as the residual correlation stays below 0.869. It holds on the measured value, with a thin margin.
- Predicted gain (-0.0004) and measured gains (-0.0007 on selection folds, -0.0010 on fresh folds) agree in sign.

Points that cut both ways: the expected gain of the average is about 0.0005 to 0.001 RMSLE, which is small against the fold-to-fold spread (std 0.013 to 0.015). It is unlikely to be visible on the leaderboard either way. The risk of the average is the linear member on out-of-range rows. The risk of CatBoost alone is giving up a small, consistent hedge.

<<DIRECTOR RULING>>

## 12. Self-check questions

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
11. As CatBoost improved from 0.11946 to 0.11807, why did the gain from averaging with Lasso halve? *The average only helps while the residual correlation stays below a break-even that depends on the accuracy gap. Here it is 0.869 (observed 0.856). A more accurate CatBoost lowers the break-even and leaves less room for the weaker Lasso. Measured gain went from -0.0014 to -0.0007.*
12. With s1 = 0.1272, s2 = 0.1190 and rho = 0.856, what weight minimises the average's error, and why use 0.5 anyway? *w* = 0.73 on CatBoost, with a predicted pooled RMSE of 0.1176 against 0.1186 at 0.5. But the optimum is estimated from the same residuals it is scored on, and the learned blend E069 measured worse than the plain average. Equal weights give up about 0.001 and have no parameter to overfit.*
13. Why is sd/sqrt(15) too small for a RepeatedKFold 5x3 difference, and what does Nadeau-Bengio change? *Repeats reuse the same rows and training sets overlap, so the 15 folds are dependent. The corrected variance is (1/15 + n_test/n_train) s^2, whose second term does not shrink with repeats. For E077 against E076 the SE rises from 0.0018 to 0.0038.*
14. Why do finalists have to be re-scored on fresh folds, and how large was the optimism? *Selection picks the best of many tries on the same folds, so the score is biased low. Fresh-fold scores were worse for all eight finalists, by 0.0036 on average (range 0.0027 to 0.0044).*
