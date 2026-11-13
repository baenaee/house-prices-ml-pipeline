# Study guide, part 2a: validation methodology

Draft by Agent A (feature-researcher) for the director to merge into `docs/STUDY_GUIDE.md`. Same audience as part 1: a third-year data-science student who must defend every decision in an interview. Function and file names are quoted from `src/houseprices/validation.py`, `registry.py`, `evaluate.py`, `config.py` and from `docs/VALIDATION_AUDIT.md` and `docs/EXPERIMENT_HISTORY.md` as read on writing. Decision numbers (D3, D14, D15, D17) and finding numbers (F1 to F28) refer to those documents.

---

## 1. The metric

### 1.1 Definition and equivalence with RMSLE

For houses i = 1..n with actual price a_i and predicted price p_i, the competition metric is

RMSLE = sqrt( (1/n) * sum_i ( log(1 + p_i) - log(1 + a_i) )^2 ).

We train on y_i = log(1 + a_i) (the function `validation.to_log_target`, which is `numpy.log1p`) and a model predicts yhat_i on that same scale. Then

RMSE on the log scale = sqrt( (1/n) * sum_i ( yhat_i - y_i )^2 ),

and when yhat_i = log(1 + p_i) the two expressions are **identical**. That is the whole equivalence: the competition metric is the ordinary RMSE of a model fitted on log1p(price). The code function is `validation.rmse_log`; `validation.rmsle_dollars` computes the metric from dollar values as a cross-check, and `validation.from_log_target` (`expm1`) converts predictions back to dollars. Using the log scale for training and scoring at once means the model optimises exactly the quantity it is judged on.

### 1.2 Why it penalises relative errors

The difference of two logarithms is the logarithm of a ratio:

log(1 + p) - log(1 + a) = log( (1 + p) / (1 + a) ).

For prices in the tens of thousands of dollars the "+1" is negligible, so this is log(p / a). A prediction that is 10% too high has error log(1.10) = 0.095 whatever the price level; a prediction 10% too low has log(0.90) = -0.105. So a $20,000 miss on a $100,000 house (20%) costs about log(1.2) = 0.18, while the same $20,000 on a $400,000 house (5%) costs about 0.049. The metric counts percentages, not dollars. Two consequences: cheap houses carry as much weight as expensive ones, and an under-prediction by a fixed percentage costs slightly more than an over-prediction by the same percentage (0.105 versus 0.095 at 10%), because the logarithm is asymmetric.

### 1.3 What an RMSE of 0.12 means in dollars

An RMSE of r on the log scale means a typical multiplicative error of e^r. For r = 0.12: e^0.12 = 1.1275 on the high side and e^-0.12 = 0.8869 on the low side, that is about +12.7% or -11.3%.

| Price level | High-side error (+12.7%) | Low-side error (-11.3%) |
|---|---|---|
| Median price, $163,000 | about +$20,800 | about -$18,400 |
| $400,000 | about +$51,000 | about -$45,200 |

Our current best linear-plus-skew result is 0.1256 (E042) and the mean baseline is 0.3987, the standard deviation of log-price. State it as "a typical error of about 13% of the price". Remember it is a root-mean-square typical error: individual houses can be far off (large expensive houses are the usual culprits), and predictions converted back with `expm1` are close to median prices, not expected prices.

---

## 2. K-fold cross-validation

### 2.1 What we estimate and what is biased

We want the expected error of the **procedure** (preprocessing plus model plus hyper-parameters) on a house it has never seen. K-fold cross-validation splits the n development rows into k parts, trains on k-1 parts, scores on the held-out part, and averages the k scores:

CV = (1/k) * sum_j RMSE_j.

Two effects control its quality.

- **Bias.** Each model is trained on (k-1)/n * n rows, fewer than the full data. Models trained on less data are slightly worse, so CV is mildly **pessimistic**. With n = 1,241 development rows and k = 5, each model trains on about 993 rows and is scored on about 248. Larger k lowers this bias (10-fold trains on about 1,117 rows) at the cost of more compute.
- **Variance.** Training sets of different folds overlap heavily (any two share 3/4 of the data when k = 5), so the k scores are positively correlated. That makes the average less stable than an average of k independent numbers would be, and larger k makes the overlap, and the correlation, larger.

Five folds is the standard compromise for 1,241 rows: low enough cost to run hundreds of configurations, bias small compared with differences between models, and enough folds to estimate a spread (decision D2). A single 80/20 split would have much higher variance.

### 2.2 Why shuffled with a fixed seed

`validation.make_cv` returns `KFold(n_splits=5, shuffle=True, random_state=42)`. Shuffling matters because the file is ordered by `Id`, and Ids can correlate with neighbourhood or sale date; unshuffled folds would each contain a different slice of the market. The fixed seed (`config.SEED`, 42) matters for a different reason: every configuration is scored on **exactly the same five partitions**. That makes comparisons paired (section 2.4) and makes every score reproducible.

### 2.3 Fold standard deviation versus standard error of the mean

Take E003 (Ridge on base): the five fold scores are 0.120, 0.185, 0.163, 0.112, 0.125. The fold standard deviation is s = 0.0316 (`CVResult.std`, with ddof = 1). It describes how much difficulty varies between folds: fold 2 contains hard houses. The **standard error of the mean** would be s / sqrt(k) = 0.0316 / 2.24 = 0.0141 if the folds were independent, which they are not (section 2.1), so even that figure understates the uncertainty. Neither quantity is the right ruler for comparing two models, because most of the fold-to-fold swing is shared by both models.

### 2.4 Paired comparisons and the one-SE rule (D14)

Because both models see the same folds, compute the per-fold difference d_j = score_A,j - score_B,j. Fold difficulty cancels, and only the difference between the models remains. The mean difference is d-bar = (1/k) * sum d_j, and its **paired standard error** is

SE_d = s_d / sqrt(k), with s_d the standard deviation of the d_j.

Example from the outlier study (D15): Ridge with Ids 524 and 1299 dropped scores +0.0044 worse than keep-all with a paired SE of 0.0053, even though the fold standard deviation is about 0.03. The paired ruler is about six times tighter than the unpaired one, which is why pairing is worth the trouble.

The registry's retain/reject/investigate decision (`experiments/feature_ablation/_ablation_common.compare_text`, ruling D14) calls a difference real only if |d-bar| exceeds one paired SE. Be precise about what this is. With k = 5 the statistic d-bar / SE_d follows a t-distribution with 4 degrees of freedom, and a value of 1 corresponds to a two-sided p-value of about 0.37. So the one-SE rule is a **noise filter**, not a significance test: it removes differences that are clearly within noise, but a surviving difference is only weak evidence. That is exactly why D14 labels all such decisions provisional and the finalists are re-checked with RepeatedKFold 5x3 on fresh folds (section 3).

---

## 3. Selection bias: the optimism of picking the best

### 3.1 The mechanism

If you score m configurations on the same folds and keep the one with the lowest CV, you pick the one that was both good **and lucky on those particular partitions**. The minimum of m noisy estimates is lower, on average, than the true error of the winning configuration. The more configurations you compare (a 136-call linear grid, 25 Optuna trials per booster) and the closer they are in true quality, the larger the optimism. The CV score of the winner is no longer an unbiased estimate of its future error.

### 3.2 What Agent D measured

| Measurement | Result |
|---|---|
| Ridge, 15-point alpha grid selected on the five selection folds | scored 0.1319 there and about 0.1340 on five fresh partitions, so optimism of about +0.002 |
| Lasso E056 | registered 0.1255 on the seed-42 folds, 0.1282 on the seed-7 folds, so +0.0027 |
| Seed-to-seed standard deviation of a CV mean | 0.0026 |
| Boosters, best of 25 trials | not measured; expected +0.002 to +0.005 |

Read this as three facts. Selection optimism is real and about 0.002 to 0.003 for the linear family. A CV mean moves by about 0.0026 simply by changing the partition seed. And therefore two configurations whose scores differ by less than about 0.003 are indistinguishable.

### 3.3 The rules that follow

1. **Differences below 0.003 between tuned models are not wins** (VALIDATION_AUDIT round 4; EXPERIMENT_HISTORY). A 0.0004 gain from a tweak is noise.
2. **Finalists are re-checked with a fresh seed** (D17): `validation.make_cv(5, 3, config.CV_SEED_FINALISTS)`, with `CV_SEED_FINALISTS = 7`. Finding F20 showed why: `RepeatedKFold(random_state=42)` has as its first repeat the identical five folds used for selection, so a "5x3" check with seed 42 would inherit the selection optimism in one third of its folds. With seed 7 none of the 15 folds equals a seed-42 fold (the largest validation overlap is 25% against 20% expected by chance).
3. **Registry decisions after selection are provisional** (F23): every best-of-grid cell is auto-labelled "retain", to be read as "retain pending the finalist check".
4. **The holdout gives one last, independent estimate** (section 4.3).

---

## 4. Leakage: a taxonomy and where each guard lives

**Leakage** means information from the data you evaluate on influences the model you evaluate. The score then promises more than the model can deliver on new houses. The table lists the kinds that matter here and the guard in this code base.

| Kind | How it would leak | Guard in this project |
|---|---|---|
| Preprocessing | median, scaler, one-hot categories, skew selection computed on all rows | everything fitted lives inside a scikit-learn `Pipeline`; `validation.cross_validate` fits a fresh `clone` of the whole pipeline on each training fold |
| Target | `SalePrice` used to build inputs | cleaning and features never read the target; a test inspects the code for `TARGET` (`tests/test_cleaning.py`) |
| Row filters | outliers removed before the split, so validation loses its hardest rows | `validation.iter_folds` applies the `train_row_filter` after the split, to the training part only (D3, D15) |
| Selection | best of many configurations judged on the folds that picked it | fresh seed for finalists (D17), the 0.003 rule, the single-use holdout |
| Holdout reuse | scoring the holdout repeatedly while tuning | `validation.score_holdout_once` and `HoldoutAlreadyUsedError`, usage log, `_assert_is_holdout_frame` (F21) |
| Stacking | meta-learner trained on in-sample base predictions | nested stacking inside `fit` (cross-reference part 3) |

### 4.1 Preprocessing inside the pipeline

Section 3 of part 1 covers the principle. The code-level guard is that the model builders (`houseprices.models:build_ridge` and the others) return a complete `Pipeline` that accepts the cleaned DataFrame, and `validation.cross_validate` calls `clone` on it per fold. `LazyPreprocessor` builds the column transformer from the training fold it receives and reads only names and dtypes when doing so. Nothing fitted survives from one fold to the next.

### 4.2 Training-row filters: only on the training side, and the fold-2 story

Dropping outliers is a modelling choice and must be evaluated like one. `validation.iter_folds` first creates the fold indices on the **full** development frame, then passes only the training rows through the filter (`keep_all`, `drop_candidate_outliers` for Ids 524 and 1299, or `drop_large_living_area(4000)`). The validation fold is never filtered, and `cross_validate` raises if a filter empties a training fold (F3, `ValueError`).

The result (D15, experiments E049 to E054): keep-all won for all three models. The mechanism is visible in the folds. Ids 1299, 692 and 1183 fall in validation fold 1 and Id 524 in fold 2. A model trained without them is still graded on them, so the linear fold standard deviation rises from about 0.023 to about 0.035. A public notebook that deletes the rows before splitting removes them from the validation side as well, deleting the hardest cases from the exam, so its score is better than the model deserves.

### 4.3 The single-use holdout

`validation.split_development_holdout` carves a 15% holdout (219 houses, stratified by price decile with `StratifiedShuffleSplit`, seed 42) and stores the ids once in `experiments/holdout_ids.json`. `load_or_create_holdout_ids` reads that file as the source of truth and never recomputes it. The development set is the remaining 1,241 houses; model selection never sees the holdout.

At the end, `validation.score_holdout_once(fitted_estimator, X_holdout, y_holdout_log, frozen_experiment_id)` scores it once and appends the event to `experiments/holdout_usage.log`. A second call raises `HoldoutAlreadyUsedError` unless `force=True`, and a forced call is logged as forced so the report must state that the holdout is no longer unbiased. Finding F21 showed that the first version of this guard was advisory, since it accepted any frame, so a mistaken call with development rows would spend the single use. The fix, `_assert_is_holdout_frame`, runs first and refuses a frame whose Id set is not the registered holdout; it is raised before the use is consumed.

### 4.4 Nested stacking

A stacking ensemble trains a meta-learner on base-model predictions. If those predictions are in-sample, the meta-learner learns to trust whichever base model memorised the training data. `NestedStackingRegressor` avoids this by building out-of-fold predictions **inside** `fit`, so `validation.cross_validate` remains the single outer protocol. Part 3 of the guide covers it; here, remember only that the leakage guard is structural (nesting inside `fit`) and that the test `test_inner_oof_rows_are_never_seen` pins it.

### 4.5 A leakage you can create after the experiments

`train.py` can fit on `rows="all"`, which includes the holdout. An artefact fitted that way must never be given to `score_holdout_once`. Finding F24 led to a guard: `train.fit_registered` refuses `rows="all"` until `validation.holdout_usage()` shows the holdout has been scored, and the artefact's sidecar records `includes_holdout_rows`.

---

## 5. Reproducibility

### 5.1 The registry record

Every experiment is one JSON line in `experiments/registry.jsonl`, written by `registry.append`, which assigns the next id (`next_id`, E001, E002, ...) when the id is empty. The `Experiment` dataclass stores identity (id, agent, title), the research protocol (hypothesis, rationale, implementation, evaluation text), the **reproducibility fields** (`builder` as "module:function", `builder_kwargs`, `feature_set`, `train_row_filter`, `cv_n_splits`, `cv_n_repeats`, `seed`), the results (`fold_scores`, `cv_mean`, `cv_std`, `fit_seconds`) and the conclusions (`interpretation`, `decision`, `tags`, `notes`, `timestamp`). `registry.write_summary` regenerates `experiments/registry_summary.csv`, sorted by `cv_mean`.

### 5.2 The config hash

`Experiment.config_hash` is the first 12 hexadecimal characters of the SHA-256 of a JSON document with sorted keys containing `builder`, `builder_kwargs`, `feature_set`, `train_row_filter`, `cv_n_splits`, `cv_n_repeats` and `seed`. It excludes titles and results. Two experiments with the same hash are the same computation, so a runner can skip a configuration already registered (`_common.find_registered`). A subtlety found later (F26): `find_registered` originally built its probe with the default seed 42, so it could not match a record registered with seed 7, and re-running a finalist created a duplicate. It now takes a `seed` argument. The lesson: a hash is only as useful as the fields that go into it, and every lookup must fill all of them.

### 5.3 Reproducing an experiment

`python -m houseprices.evaluate --experiment E003` calls `evaluate.run_registered`. It reads the record, resolves the builder string with `data.resolve_callable`, rebuilds the estimator from the stored kwargs, rebuilds the data with `data.load_development_data` and the folds with `validation.make_cv` using the stored repeats and seed, runs `cross_validate`, and compares the new fold scores with the registered ones (tolerance 2e-4). The report says `reproduces_registered_scores: true` or false, and prints `n/a` if you override repeats or seed, because then the protocol differs. Example: E042 reproduced 0.12562 against 0.125622 registered. The environment is pinned in `requirements.txt` (pandas 2.3.3 and so on); finding F14 notes that the record does not store library versions or the git commit, so that part of reproducibility rests on the pinned file (the train.py sidecar does record them).

### 5.4 Why `train.py` and `predict.py` are not CV evidence

CV evidence is a score on rows the model did not train on. `train.py` fits a registered configuration on a chosen set of rows (development rows or all rows) and `predict.py` writes predictions for the competition test file, which has no prices. Neither produces a held-out score. A model fitted on all rows and then "evaluated" on the same rows would only show how well it memorises. The only valid numbers are the CV scores in the registry, the finalist RepeatedKFold with the fresh seed, and the one logged holdout call.

---

## 6. What the audit rounds found, and why independent review matters

The auditor (Agent D) did not write the code it reviewed, so it could test what the authors assumed. The table lists the findings as recorded in `docs/VALIDATION_AUDIT.md`. Severity is the auditor's; status is as read from the code and tests on writing, and the audit file is the source of truth for any later change.

| ID | Severity | Finding | Status |
|---|---|---|---|
| F1 | Medium | `is_single_pass` was always true (`RepeatedKFold` has no `n_splits`), so out-of-fold predictions of repeated CV were overwritten instead of being NaN | fixed |
| F2 | Medium | NaN predictions scored silently; a broken experiment sank to the bottom of the summary | fixed (raises on non-finite) |
| F3 | Low | a row filter that left no training rows was not detected | fixed (raises) |
| F4 | Low | holdout JSON trusted without checking ids against the data | fixed |
| F5 | Low | dictionary spelling key `CmentBd` should be `CemntBd` | fixed |
| F6 to F8 | Info | duplicate ids allowed in an append-only log; import needs `train.csv`; filter name must encode its threshold | noted, F8 closed for integers |
| F9 | High | the feature table had no `Id`, so the outlier filter raised `KeyError` | fixed (`Id` re-attached in `data.py`) |
| F10 | Low | dry-run data could be registered as a real experiment | fixed (raises in dry-run mode) |
| F11 | Low | text-column detection broke under pandas 3 | fixed |
| F12 to F15 | Info | non-integer filter thresholds, silent fallbacks in LotFrontage options, no library versions in the record, flags chosen by inspecting training data | noted |
| F16 | Low | a stacking test did not detect the classic in-sample meta-learner leak | test strengthened |
| F17 | Low | optimistic diagnostics could be hand-entered into the registry | guard added |
| F18, F19 | Info | fixed blend weights chosen from the same scores are optimistic; `passthrough` unreachable | noted |
| F20 | Medium | the finalist 5x3 check shared its first repeat with the selection folds | fixed (D17, seed 7) |
| F21 | Low | `score_holdout_once` accepted any frame | fixed (`_assert_is_holdout_frame`) |
| F22, F23 | Info | tuning cap not enforced in code; best-of cells auto-labelled retain | noted |
| F24 | Medium | `train.py --rows all` allowed before the holdout is scored | fixed in code (`train.fit_registered` refuses `rows="all"` until the holdout is scored; test `test_fit_rows_all_is_refused_until_the_holdout_is_scored`) |
| F25 | Low | `predict.py` accepts overflowing log predictions that become `inf` prices | fixed in code (`predict_test` now checks that prices are finite) |
| F26 | Low | finalist duplicate detection can never match a seed-7 record | fixed in code (`find_registered` takes a `seed` argument and `run_finalists.py` passes it) |
| F27, F28 | Info | registry append is not atomic or locked; artefacts are pickles | noted |

The lessons are practical. F9 was a High-severity defect hidden behind a comment that was simply false ("X includes Id"). F1 passed every casual read because the code looked right. F20 is a statistical, not a coding, defect: the code did exactly what it said, and the protocol was still subtly biased. F21 shows that a guard that does not check what it guards is advisory. The common pattern is that authors read what they meant, while a reviewer runs what is written. This project's rule is therefore: every binding protocol rule gets an executable test written by someone other than its author, and a finding that is not yet fixed is pinned by a strict expected-failure test so that fixing it forces the marker to be removed.

---

## Self-check questions

1. Show that RMSE on log1p of the price equals the Kaggle RMSLE, and say which function in the code computes it.
2. A model's RMSE is 0.12. At the median price of $163,000, roughly what are the typical high-side and low-side errors? Why are they not symmetric?
3. Why is K-fold CV mildly pessimistic, and why are the k fold scores correlated?
4. Why do we shuffle the folds, and why do we fix the seed?
5. The fold standard deviation of E003 is 0.0316. Why is that the wrong number to judge a 0.003 difference between two models, and what do we use instead?
6. State the one-SE rule precisely and say honestly what it is and is not statistically.
7. Explain selection bias. What did Agent D measure for the Ridge grid and for Lasso E056, and what rule follows?
8. Why does `make_cv(5, 3, 42)` not give an independent finalist check, and what was done about it?
9. Explain where in the code a training-row filter is applied, and why dropping Ids 524 and 1299 before the split would flatter the score. What did the honest protocol show?
10. What does the config hash contain, what does `evaluate --experiment E003` do, and why does a `train.py` fit not count as validation evidence?

## Answers

1. The competition computes sqrt of the mean of (log(1+p) - log(1+a))^2. Training on y = log1p(price) and predicting yhat = log1p(p) makes this exactly sqrt of the mean of (yhat - y)^2, the ordinary RMSE on the log scale. `validation.rmse_log` computes it, and `rmsle_dollars` computes it from dollar values as a check.
2. About 12.7% too high or 11.3% too low, so roughly +$20,800 or -$18,400 at $163,000. The difference of logs is the log of a ratio, and the logarithm is asymmetric: e^0.12 = 1.1275 but e^-0.12 = 0.8869.
3. Each model trains on only (k-1)/k of the data, so it is a little worse than a model trained on everything, which makes CV pessimistic. The training sets of different folds overlap by (k-2)/(k-1) of their rows, so the fold errors are positively correlated and the mean is less stable than an average of independent scores.
4. The file is ordered by Id, which can correlate with neighbourhood or date, so shuffling prevents folds from being different slices of the market. A fixed seed makes every configuration use identical partitions, which allows paired comparison and exact reproduction.
5. Most of the fold-to-fold variation (hard folds versus easy folds) affects every model equally, so it hides nothing about the difference between two models. We compute the per-fold differences, take their mean and the paired standard error s_d divided by the square root of k, which is several times smaller (for example 0.0053 in the outlier study).
6. A difference is called real if the absolute mean per-fold difference exceeds its paired standard error. With 5 folds that is a t-statistic of 1 on 4 degrees of freedom, a two-sided p-value of about 0.37. So it is a noise filter that discards differences clearly within noise, not a significance test, which is why decisions are provisional (D14).
7. Choosing the best of many configurations on the same folds picks one that was also lucky on those partitions, so its CV is optimistic. Ridge's 15-point grid scored 0.1319 on the selection folds and about 0.1340 on fresh ones (+0.002); Lasso E056 scored 0.1255 and 0.1282 on seed-7 folds (+0.0027); the seed-to-seed SD of a CV mean is 0.0026. Hence differences below about 0.003 between tuned models are not wins, and finalists are checked on fresh folds.
8. The first repeat of `RepeatedKFold(random_state=42)` is the identical five folds used for selection, so one third of the check inherits the selection optimism (finding F20). The fix, D17, uses `config.CV_SEED_FINALISTS = 7`; none of the 15 seed-7 folds equals a seed-42 fold.
9. `validation.iter_folds` creates fold indices on the full frame and applies the filter to the training indices only; the validation fold stays complete. Dropping the houses before splitting removes the hardest cases from validation as well, so the exam gets easier. Under the honest protocol, keep-all was best for all three models (D15), and the linear fold standard deviation rose from about 0.023 to about 0.035 when the houses were dropped from training only.
10. The first 12 characters of a SHA-256 over builder, builder kwargs, feature set, row filter, folds, repeats and seed; titles and results are excluded. `evaluate` rebuilds the estimator, data and folds from the record, reruns the CV and compares the fold scores to the registered ones within 2e-4, reporting whether they reproduce. A `train.py` fit produces a model, not a score on unseen rows, and an all-rows artefact has even seen the holdout, so only the registered CV scores, the fresh-seed finalist check and the single logged holdout call are evidence.
