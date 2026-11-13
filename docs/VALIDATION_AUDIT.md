# Validation audit (Agent D)

## 1. Scope and method

Independent review of the binding methodology (decisions D1-D6, requirements R1-R7). Round 1 covers `src/houseprices/config.py`, `validation.py` and `registry.py`. Method: line-by-line reading, executable tests in `tests/test_validation.py` (synthetic frames with Id, GrLivArea, SalePrice, plus the real `train.csv` for the holdout), and probing sklearn 1.7.2 behaviour directly. Pipeline code is read-only for the auditor; bugs are reported, not fixed.

Test result (houseprices env, Python 3.12): `19 passed, 3 xfailed`. The three xfails are strict and pin findings F1-F3; they flip to failures when the code is fixed, which is the signal to remove the marker.

> Status: round 1 complete (F1-F8 closed or noted). Round 2 (Phase A pipeline: cleaning, features, preprocess, models, data, runner) complete on 2026-11-13, see section 5. Phase A SIGN-OFF: approved, baselines E001-E004 may run. Ensembles, outlier-policy and tuning scripts and the final pipeline remain for later rounds.

## 2. Round 1 findings

| ID | Severity | Location | Finding | Status |
|---|---|---|---|---|
| F1 | Medium | validation.py:179-180, 192-193 | `is_single_pass` is always True. `RepeatedKFold` has no `n_splits` attribute, so `getattr(cv, "n_splits", n_splits_total)` returns the default and the comparison is trivially true. For `make_cv(n_repeats=3)` the OOF array is overwritten repeat after repeat (last repeat wins) instead of being NaN as `CVResult` documents. Fold scores are unaffected, but any later code (stacking meta-learner, residual analysis) that trusts `oof_pred` from a repeated run would use one arbitrary repeat. Failing example: `cross_validate(est, X, y, make_cv(n_repeats=3)).oof_pred` has no NaN. Fix: `is_single_pass = n_repeats == 1`, e.g. `isinstance(cv, KFold)`. | Open (test_repeated_cv_oof_is_nan_as_documented) |
| F2 | Medium | validation.py:190 | NaN predictions are not rejected. `rmse_log` returns NaN, `cv_mean` becomes NaN and is appended to the registry. `write_summary` sorts with `na_position="last"`, so a broken experiment disappears quietly. Also `np.mean` of scores with one NaN poisons everything, which is visible, but there is no error at the point of failure. Fix: raise `ValueError` if `not np.isfinite(pred).all()`. | Open (test_nan_predictions_should_raise) |
| F3 | Low | validation.py:157-160 | No guard when the filter leaves zero (or very few) training rows. A strict estimator fails inside sklearn; a tolerant one (mean of nothing) yields NaN. Fix: raise if `len(tr) == 0`, and record `n_train_rows_per_fold` (already stored) in the registry. | Open (test_empty_training_after_filter_fails_loudly) |
| F4 | Low | validation.py:87-92 | `load_or_create_holdout_ids` trusts an existing JSON without checking that the ids exist in `train` or that the count equals `round(0.15*n)`. A stale or hand-edited file would silently shrink or corrupt the development set. Also creation depends on row order of the frame passed in, so the JSON, never recomputation, must be the source of truth (the code does this correctly). Suggest asserting `set(ids) <= set(train.Id)` on load. | Open |
| F5 | Low | config.py:116 | `DICTIONARY_TO_DATA_VALUES["Exterior2nd"]` contains `"CmentBd": "CmentBd"` (a no-op). The dictionary spelling is `CemntBd` (data_description.txt:209, 229), the data spells `CmentBd`. The key should be `"CemntBd"`. Harmless unless the mapping is applied programmatically, then Exterior2nd would be left unmapped for that level. | Open, for Agent A |
| F6 | Info | registry.py:93-96, 107-113 | `next_id` takes the max numeric suffix, fine. `append` does not reject a duplicate explicit id, and `get_experiment` silently returns the last duplicate. Acceptable for an append-only log but re-running an experiment under the same id would overwrite history logically. Not a validation bug. | Note |
| F7 | Info | config.py:32 | `PROJECT_ROOT` is resolved at import and requires `train.csv`; importing the package fails outside the project. Acceptable. | Note |
| F8 | Info | registry.py | Registry records `train_row_filter` only by name. For `drop_large_living_area(threshold)` the factory sets `__name__` to include the threshold, so the name is unique; evaluate.py must resolve it the same way to satisfy R6. | To check in round 2 |

Verified without findings:

- `rmse_log` equals the hand computation (errors 0.1, -0.2, 0.3 give sqrt(0.14/3) = 0.2160247) and equals `rmsle_dollars` on the same data; `to_log_target`/`from_log_target` are an exact inverse pair.
- `make_cv`: `KFold(shuffle=True, random_state=seed)` is identical across calls, differs for seed 43, partitions all rows exactly once; `RepeatedKFold` is deterministic (15 splits).
- `make_holdout_ids`: deterministic, sorted, 219 ids for the real 1,460 rows, no duplicates, decile shares within 2 percentage points of the overall shares; `split_development_holdout` returns a disjoint partition and reuses the stored file.
- `iter_folds`: the filter receives only training rows, shrinks only `tr`, validation positions are identical for `keep_all`, `drop_candidate_outliers` and `drop_large_living_area`; a wrong-shaped mask raises.
- `cross_validate`: clones per fold (original never fitted), never predicts on rows it fit on, a Y-memorising estimator scores 0 on its own training rows but exactly the same as the mean model in CV, so the runner scores unseen rows only. Ids 524 and 1299 are never in any training fold under `drop_candidate_outliers` yet are still scored when they fall into a validation fold.
- Subtle point, acceptable: policies that drop the two outliers make the validation score include them as hard cases; this is intended (D3).

Not yet created: `experiments/holdout_ids.json` does not exist, so R4 cannot be fully verified (only the generator).

## 3. Checklist R1-R7

| Req | Status | Evidence / note |
|---|---|---|
| R1 log1p target, RMSE on log scale, expm1 inverse | Verified | Tests for rmse_log, rmsle_dollars, round trip. Round 2: `load_development_data` builds `y_log` with `to_log_target(dev[SalePrice])`, aligned by Id; `_common.run_cv` and `evaluate` pass `dev.y_log`. |
| R2 fitted preprocessing inside training folds | Verified (Phase A) | Cleaning has no fitted statistic (AST probe). All learning is in `LazyPreprocessor`, `NumericMedianImputer`, `GroupMedianImputer`; sklearn `cross_validate(return_estimator=True)` shows per-fold medians equal the training-fold medians and differ between folds. Optional `SkewLog1pTransformer` and `KNNLotFrontageImputer` (default off) are fit-only too but not used by E001-E004. |
| R3 filters after split, training part only, identical validation index | Verified (runner level) | Tests d. Still to check: any future filter that uses statistics across training rows must not see validation rows (the interface guarantees this). |
| R4 holdout carved once, deterministic, seed 42, decile-stratified | Partly verified (round 2: F4 fix confirmed; `holdout_ids.json` still absent and will be created by the first real run; `_common.run_cv` never reads the holdout frames) | Generator verified. JSON not yet written; no tuning-time access control exists (nothing prevents loading the holdout frame). Re-check usage log at the end. |
| R5 nested CV for stacking | Not yet audited | `ensemble.py` exists (Agent C, synthetic tests only); audit in a later round. F1 is fixed, so OOF from single-pass `cross_validate` is well defined. |
| R6 reproducibility via `evaluate --experiment` | Verified | Executable probe registers a Ridge run in a temporary registry and `evaluate.run_registered` reproduces the fold scores (same data entry point, seed, filter resolution). |
| R7 originals untouched, no SalePrice in features, no test info in fits | Verified (Phase A) | AST probes: cleaning and features never reference SalePrice or any fitting call; `clean` is row-wise (subset equals subset of full), so train and test cannot influence each other; train and test are never concatenated. Original-file hashes to be re-confirmed in the final round. |

Cross-cutting item (added by director): dtype checks must use `select_dtypes(include=["object", "string"])` so behaviour is identical under pandas 2 and 3. Status: not yet applicable; to be checked in cleaning/preprocess rounds. Environment: pandas 2.3.3, scikit-learn 1.9.1; the validation tests were re-run there with the same result as on sklearn 1.7.2 (19 passed, 3 xfailed), so finding F1's root cause (RepeatedKFold has no `n_splits`) holds on 1.9.1 as well.

## 4. Open questions for the director

1. Should `cross_validate` reject non-finite predictions (F2) and use a correct single-pass test (F1)? Recommended yes, both are two-line changes by the code owner.
2. How will holdout access be controlled during tuning? Suggest a function that logs each call to `experiments/holdout_usage.log` and refuses a second scoring call.
3. Selection bias: with many experiments compared on the same 5 folds, small differences below the fold standard deviation are noise. Proposal: only retain changes that win on RepeatedKFold 5x3 or by more than one paired standard error.
4. `drop_large_living_area(4000)` is not equivalent to the two-Id list. On train.csv it selects four houses: Ids 524 and 1299 (low price for size) but also 692 and 1183 (prices 755,000 and 745,000, consistent with their size). Those two are legitimate extreme-value houses, so this policy removes real signal and should be registered as a distinct experiment, not as a proxy for the Id list. Rule-based variants should use the area-versus-price residual, not area alone.

## 5. Round 2 - Phase A pipeline audit

Date 2026-11-13. State audited: working tree with the director's `data._with_id` fix, `cleaning.clean_train_test` calling `add_anomaly_flags`, the `anomaly_flags` feature group, and Agent B's `_common.py` with dry-run mode. `preprocess.py` gained optional `SkewLog1pTransformer` and `KNNLotFrontageImputer` (default off) while the audit was running; they were probed lightly and are NOT used by E001-E004. Executable probes: `tests/test_audit_round2.py` (48 passed, no xfail). Full suite: 165 passed. The probes never create the real holdout or registry files (paths are redirected to temp directories).

### 5.1 Findings

| ID | Severity | Location | Finding | Failing example | Suggested fix | Status |
|---|---|---|---|---|---|---|
| F9 | High | features.py:54-62 (`group_base`), data.py `load_development_data` | `group_base` drops `Id`, so the feature table had no `Id`; `validation.drop_candidate_outliers` selects rows by Id and raised KeyError. The comment "X includes Id" was false and `describe()` subtracted a non-existent column. | `drop_candidate_outliers(dev.X, dev.y_log)` raised KeyError 'Id' | Re-attach Id at the data entry point (models already drop it). | Fixed by director (`_with_id`), verified by me: Id is the first column of X, X_holdout and X_test; the filter removes exactly Ids 524 and 1299; Id never reaches an estimator (`LazyPreprocessor` and `NumericMedianImputer` drop it, output feature names checked). Tests: `tests/test_data.py` and `test_base_group_excludes_id_by_design_and_data_reattaches_it`. |
| F10 | Low | experiments/models/_common.py:21-41, 70-100 | New dry-run mode (`set_dry_run`) makes `get_data` return the first N rows of the full table with no holdout split, but `run_and_register` has no guard. If a script calls `set_dry_run(n)` and registers, a truncated-data score enters the append-only registry as a real experiment. No current script does this. | `set_dry_run(120)` then `run_and_register("E902", ...)` appends a record | Raise in `run_and_register` when `_DRY_RUN_ROWS` is not None. | Fixed (`run_and_register` now raises RuntimeError in dry-run mode, `_common.py:76`); verified by `test_dry_run_results_must_not_be_registered`, which flipped from strict xfail to pass |
| F11 | Low | models.py `CatBoostNative.fit` | Text columns are detected with `dtype == object` or `str(dtype) in ("category","string")`. Under pandas 3 the default text dtype prints as "str", so categoricals would be missed. Correct today (pandas pinned 2.3.3), but violates the pandas 2/3 rule. | pandas 3: `str(pd.Series(["a"]).dtype)` is "str" | Use `not pd.api.types.is_numeric_dtype(X[c])`, as `preprocess.split_columns` does. | Fixed by Agent B (`models.py:153` now uses `not is_numeric_dtype`); verified by reading the code, behaviour covered by `test_catboost_native_fill_text_learns_nothing` |
| F12 | Info | data.py `get_train_row_filter` | Threshold parsed with `float`, factory name built with `int(threshold)`; a non-integer threshold would resolve to a filter with a different name. Integer thresholds (all planned) are consistent. Closes F8 for integers. | `get_train_row_filter("drop_GrLivArea_gt_4000.5").__name__` is `drop_GrLivArea_gt_4000` | Parse as int or format the name with `:g`. | Open, note |
| F13 | Info | preprocess.py `make_preprocessor` | With `lotfrontage="knn"` and a missing predictor column (or `neighborhood_median` without Neighborhood), LotFrontage silently falls back to the plain numeric median. | `make_preprocessor(X.drop(columns="LotArea"), lotfrontage="knn")` | Raise instead of falling back. | Open, note |
| F14 | Info | registry.py `Experiment` | The record stores seed, builder, kwargs, folds and filter but not library versions or the git commit, so R6 relies on the pinned `requirements.txt`. | n/a | Add `git_commit` and environment fields later. | Open, note |
| F15 | Info | config.py `CONSTANT_IN_TRAIN_FLAGS` | The dropped flags were chosen by inspecting the training data (always False). This is a design-time choice, not a fitted statistic, but test-only anomalies (Id 2121, GarageYearUnknown, MiscFeatureConflict) are invisible to models. | n/a | Document in the report. | Open, note |

Round 1 re-check: F1-F3 stay fixed (ordinary tests). F4 fixed: the holdout load rejects unknown and duplicate ids (`test_f4_holdout_file_validated`). F5 fixed: the dictionary key is `CemntBd` mapping to the data spelling `CmentBd` and the old no-op key is gone (`test_f5_exterior2nd_key_is_cemntbd`). F8 closed for integer thresholds (see F12).

### 5.2 Verified without findings (evidence)

1. cleaning (a): AST probe finds no reference to "SalePrice" or `config.TARGET` outside docstrings (`test_cleaning_never_reads_target_or_fits_statistics`, plus the older test in `test_cleaning.py`).
2. cleaning (b): the same probe finds no call to median, mean, mode, std, quantile, groupby, transform, interpolate, ffill, bfill, rank, concat, merge or fit; every `fillna` argument is a literal constant. Fills are constant labels or 0 for bath counts.
3. cleaning (c): `clean(df.iloc[sample])` equals `clean(df).iloc[sample]` for 300 random rows of train and of test, so no rule uses other rows. `clean_train_test` calls the same `clean` on each table separately and never concatenates them. Columns are identical except SalePrice.
4. cleaning (d): B04 is a condition on 11 basement fields; it selects no train row and exactly test Id 2121. The Id appears only in docstrings, never in code constants.
5. cleaning (e): the `rule_c01` docstring says RECONSTRUCTION and "NOT recovered code"; `config.py` also labels it reconstructed.
6. cleaning (f): the input frames equal deep copies taken before `clean`; Id order and the SalePrice series are unchanged.
7. cleaning (g): `clean_train_test` output minus the three anomaly columns equals `clean` output exactly; test Id 2593 keeps GarageYrBlt 2207 and is flagged; Ids 2550 and 1877 are flagged.
8. cleaning (h): cleaning.py has no dtype logic; the post-clean check uses `select_dtypes(include=["object","string"])`. `preprocess.split_columns` uses `is_numeric_dtype`, identical under pandas 2 and 3. Only `CatBoostNative` is not (F11).
9. features: all seven groups are row-wise (a 200-row subset build equals the same rows of the full build); no fit calls or target references; `quality_to_number` maps Unknown, NaN and unrecognised labels to NaN, absence labels to 0, Po..Ex to 1..5; the duplicate guard raises ValueError and an unknown group raises KeyError; base excludes SalePrice, Id, constant flags and anomaly flags; the anomaly group is 0/1 integers and enters only when requested.
10. preprocess: `GroupMedianImputer` fitted on two different frames gives different group and global medians; `transform` on new rows uses the training medians, falls back to the training global median for unseen or all-NaN groups and does not change fitted state. `make_preprocessor` gives the same structure for the full frame, a shuffled copy and an empty frame (names and dtypes only). `_as_text` keeps NaN. The OneHotEncoder ignores unseen categories. MSSubClass, MoSold, YrSold route to the categorical branch. RobustScaler is present only when `scale=True`. Repeated transforms are identical.
11. models: under sklearn `cross_validate(return_estimator=True)` with 5 folds, each fitted numeric median vector equals the independently computed training-fold median and differs between folds; the LotFrontage NAmes median equals the training-fold value and differs between folds; Id is absent from the output features; the template estimator stays unfitted. `NumericMedianImputer` stores the training median and does not change on transform. `scaler="standard"` replaces RobustScaler. `CatBoostNative._fill_text` fills NaN with the constant "Unknown" and changes no fitted attribute. All eight builders construct with no data; seeds are 42 and thread counts fixed at 8. Mean and OLS pipelines predict finite values.
12. data: features are row-wise, and building on the full cleaned train then dropping holdout rows equals the entry point's X (`test_holdout_split_then_features_equals_features_then_split`); dev and holdout Ids are disjoint and total 1,460; `y_log` equals `log1p(SalePrice)` aligned by Id. `_common.run_cv` and `run_and_register` never reference holdout frames. `drop_GrLivArea_gt_4000` resolves to a filter of the same name that keeps GrLivArea <= 4000.
13. runner: `run_and_register` refuses a duplicate id; the record holds builder, kwargs, feature set, filter, splits, repeats, seed, fold scores, mean and hash. `evaluate.run_registered` reproduces the registered fold scores of a Ridge run with the outlier filter, using the same entry point and seed.
14. Optional components: see 5.4 (full audit, no findings).

### 5.4 Addendum: fitted transformers and Agent A's runners (scope added by the director)

`SkewLog1pTransformer` (preprocess.py ~50-90), `KNNLotFrontageImputer` (~100-122), `make_preprocessor(skew_log1p=..., lotfrontage="knn")`, the new tests in `tests/test_features.py`, and `experiments/feature_ablation/{_ablation_common,run_ablation,run_lotfrontage}.py`. No new findings above Info. Evidence:

1. Skew: the column list (`log_mask_`) is computed in `fit` from the rows given; `y` is ignored. In a 5-fold `cross_validate(return_estimator=True)` with `skew_log1p` and `knn`, each fold's `skewed_columns_` equals the independently computed set of columns with training-fold skew > 0.75 and minimum >= 0. `transform` of one row equals the same row inside a batch, so unseen rows use only the fitted state. Unseen negatives are clipped to 0, NaN stays NaN, all-NaN and constant columns are never selected.
2. KNN: `fit` stores standardised training rows only; `y` is ignored. Changing the LotFrontage values of other validation rows does not change any imputed value, so validation rows never join the donor pool. Observed LotFrontage values pass through unchanged. NaN in a predictor (LotArea set to NaN) yields a finite fill through `nan_euclidean`. Fits on two disjoint samples give different fills for the same rows. Batch equals single-row transform.
3. Order: lotfrontage branch is `knn` then `scale` (scaling after the fill); numeric branch is `skew`, `impute`, `scale`. LotFrontage is excluded from the numeric branch when the KNN or group imputer takes it.
4. Defaults: `make_preprocessor` with default arguments contains neither new component, so E001-E004 are unaffected.
5. `tests/test_features.py` new tests cover selection by skew and minimum, fold dependence, unseen rows, threshold edges (strictly greater), all-NaN column, and option wiring. They are sound; none touch the holdout. Note the anomaly-flag test confirms the three flags sum to 0 in train, so `base+anomaly_flags` cannot differ from `base` in CV (Agent A documents this and excludes it from the default list).
6. Runners: the three scripts reference no holdout frames, no `load_development_data` and no direct `cross_validate`; they go through `_common.run_cv`, `_common.run_and_register` (with a precomputed result) and `_common.find_registered`. A config with an existing config hash is reused, not re-run (probe: the second `Runner.get` returns the same id, registry still has one line). `--dry-run` fits once on 150 rows, writes nothing and now cannot register (F10 fix). `paired_stats` matches the hand computation.
7. Info only: decisions in the registry (retain/reject) are generated by a one-paired-SE rule on the same 5 folds. That is noisy for differences near 0.001; treat "retain" as provisional until the RepeatedKFold 5x3 check on finalists (open question 3, round 1). With `lotfrontage="knn"` the missing-predictor fallback is F13.

### 5.3 Notes for the director

- `experiments/holdout_ids.json` does not exist yet. The first real run creates it deterministically; commit it right after E001.
- The `tests/test_models.py` clone failures mentioned in the brief did not appear in the final full run.
- `ensemble.py` still needs its own audit round before experiments depend on it.

Phase A SIGN-OFF: approved, baselines E001-E004 may run.

## 6. Round 2b and Round 3

### 6.1 Round 2b: `SkewLog1pTransformer`, `KNNLotFrontageImputer`, feature-ablation runners

Full review is in section 5.4 (done during round 2 at the director's request): fit-only learning, per-fold column selection, donor pool limited to training rows, NaN predictors, scaling after the KNN fill, runners use only `_common` entry points, never touch the holdout, skip duplicate config hashes, dry-run cannot register (F10 fixed). Remaining Info items: F13 (silent fallback if a KNN predictor is missing; harmless because the ablation uses the full base set) and the one-paired-SE decision rule being provisional.

Round 2b SIGN-OFF: approved, the `--skew` ablation pass and the `lotfrontage="knn"` comparison may run.

### 6.2 Round 3: `ensemble.py` and `tests/test_ensemble.py`

Audited against binding correction 1. Probes: `tests/test_audit_round2.py`, round 3 block (7 tests, all passing).

| ID | Severity | Location | Finding | Failing example | Suggested fix | Status |
|---|---|---|---|---|---|---|
| F16 | Low | tests/test_ensemble.py `test_nested_stack_is_not_fooled_by_a_memorising_base` | The test is not sensitive to the most important bug. With a single memoriser base, a stack whose meta-learner is trained on IN-SAMPLE base predictions (the classic leak) scores the same as the honest nested stack, because the meta weight on the memoriser is irrelevant when the base returns the mean for unseen rows. Both pass the `> 0.8 * std` assertion. The OOF test (`test_inner_oof_rows_are_never_seen...`) does catch leaky OOF generation, so the code is protected, but the meta-training-data leak has no test in Agent C's file. | `_LeakyStack` mutant on `[Memoriser]` scores about std(y), like the honest stack | Add a two-base test: memoriser plus Ridge, assert nested is close to Ridge-only and the leak is much worse. My probe `test_leaky_mutant_is_caught_by_a_two_base_memoriser_probe` does this: honest within 15% of Ridge alone, leaky mutant more than 3x worse. | Open (covered by my probe; not blocking) |
| F17 | Low | ensemble.py `optimistic_oof_stack_score`, `in_sample_stack_score`; registry.py `append` | The optimistic diagnostics return a plain float and are labelled in name and docstring. They cannot flow through `_common.run_and_register` (it needs a CVResult; probe passes a float and gets AttributeError, registry untouched), and they are not registry builders, so `evaluate` cannot run them. Missing guard: nothing stops someone building `registry.Experiment(cv_mean=<optimistic float>)` by hand, and `append` has no check. | `registry.append(Experiment(..., cv_mean=in_sample_stack_score(...)))` succeeds | In `registry.append`, require `fold_scores` with `cv_n_splits` entries per repeat, and add a `tags` rule rejecting an `optimistic` tag on `retain` decisions. Optional. | Open, not blocking |
| F18 | Info | ensemble.py `build_blend(weights=[...])` | Fixed weights are not learned, so no leakage in the stack itself, but weights chosen by looking at the same development-fold OOF scores (for example from earlier experiments on the same folds) make the blend's CV score optimistic. | n/a | Use `weights=None` (learned in fit) for reported scores; fixed weights only if specified before looking at scores, or equal weights. | Note |
| F19 | Info | ensemble.py | `passthrough=True` casts X with `np.asarray(X, float)`, which fails on the project's mixed-type frames. Not reachable: `build_stack` defaults to False and the director decided passthrough=False only. File starts with a UTF-8 BOM, harmless to Python. | n/a | None needed. | Note |

Verified without findings (evidence):

1. Base models fit on outer-train only: a recording base shows 6 fits per outer fold (5 inner plus 1 refit), every fit set is a subset of the outer-train rows and disjoint from the outer validation rows (`test_nested_stack_outer_validation_rows_never_touch_any_fit`).
2. Meta-learner trained on inner OOF of the outer-train partition: for each outer fold, `inner_oof_` equals `inner_oof_matrix` recomputed on outer-train alone, and the fitted meta coefficients equal a Ridge fit on that matrix (`test_nested_stack_meta_is_fit_on_inner_oof_of_outer_train_only`).
3. Bases refit on the whole outer-train partition: fitted base coefficients equal a fresh Ridge on all outer-train rows.
4. Outer validation fold scored once with the refit bases and the meta-learner, never a row used for the meta fit: `cross_validate` calls `fit` on the training indices and `predict` on the validation indices only; `fit` never receives them (point 1).
5. `BlendingRegressor(weights=None)`: per outer fold, `weights_` equals `fit_blend_weights` on inner OOF of that fold's training rows; they sum to 1 and are non-negative; fixed weights learn nothing.
6. Single outer protocol: `NestedStackingRegressor` and `BlendingRegressor` are clone-safe estimators; nesting is inside `fit`; `validation.cross_validate` and sklearn's cross_validate both work. Training-row filters shrink the outer-train rows before `fit`, so filtered rows never enter the inner CV (Agent C's `test_filter_applies_to_training_side_only`).
7. Memoriser OOF test is sensitive to leaky OOF generation: an in-sample OOF would return y exactly and fail the `> 0.5 std` assertion.
8. R6: a stack registered through `run_and_register` (two Ridge bases, Ridge meta) reproduces through `evaluate.run_registered` (`test_stack_registers_and_reproduces_through_evaluate`).
9. Optimistic diagnostics cannot be registered through the runner (F17 describes the remaining manual path).

Round 3 SIGN-OFF: approved, ensemble experiments may run (F16-F19 are non-blocking).

### 6.3 Status update 2026-11-13

F10 and F11 verified fixed (see section 5.1). E001-E004 are registered and E003/E004 reproduce through `evaluate`; `experiments/holdout_ids.json` exists. All audit probes redirect the registry and holdout paths to temporary directories, so they never write to the real registry.
