# Validation audit (Agent D)

## 1. Scope and method

Independent review of the binding methodology (decisions D1-D6, requirements R1-R7). Round 1 covers `src/houseprices/config.py`, `validation.py` and `registry.py`. Method: line-by-line reading, executable tests in `tests/test_validation.py` (synthetic frames with Id, GrLivArea, SalePrice, plus the real `train.csv` for the holdout), and probing sklearn 1.7.2 behaviour directly. Pipeline code is read-only for the auditor; bugs are reported, not fixed.

Test result (houseprices env, Python 3.12): `19 passed, 3 xfailed`. The three xfails are strict and pin findings F1-F3; they flip to failures when the code is fixed, which is the signal to remove the marker.

> Open at the 2026-11-13 pause: F27 (non-atomic registry append, accepted limitation under the single-writer rule), F28 (pickled artefacts, note), F22/F23 (info: 3 h cap not enforced in code, automatic retain decisions are provisional); no blocking concern. Not yet audited: Stage 2 tuning scripts and any new experiment runners added after round 5.

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

## 7. Round 4 - outlier-policy, linear-grid and Optuna scripts, holdout guard

Date 2026-11-13. Scripts were read, never run; all probes redirect registry, holdout-usage and holdout-id paths to temporary directories (the real registry stayed at 57 lines during the audit). Probes: `tests/test_audit_round2.py`, round 4 block (16 tests plus 1 strict xfail). Full suite: 191 passed, 1 xfailed.

### 7.1 Findings

| ID | Severity | Location | Finding | Failing example | Suggested fix | Status |
|---|---|---|---|---|---|---|
| F20 | Medium (gates the finalist step C8, not the registry) | validation.py `make_cv`, config `CV_N_REPEATS_FINALISTS` | `make_cv(5, 3, 42)` is `RepeatedKFold(random_state=42)` and its FIRST repeat is identical to `KFold(5, shuffle, random_state=42)`, the five folds every configuration was SELECTED on. A "5x3 finalist check" therefore re-uses one third of the selection folds, so it is not an independent bound on selection bias. | `list(make_cv(5,3,42).split(X))[:5] == list(make_cv(5,1,42).split(X))` is True (`test_repeated_kfold_finalist_check_shares_its_first_repeat_with_selection_folds`) | Run the finalist check with a different seed (for example `make_cv(5, 3, 7)`, verified to differ), or report repeats 2-3 separately from repeat 1. | Fixed by director (D17: `config.CV_SEED_FINALISTS = 7`, `evaluate --seed`); verified: none of the 15 seed-7 folds equals a seed-42 fold, maximum validation overlap 25% (random expectation 20%); `evaluate --experiment E056 --repeats 3 --seed 7` reports n/a for reproduction and wrote nothing. Fresh-fold score of E056 0.1282 vs registered 0.1255 (optimism +0.0027, consistent with 7.3). |
| F21 | Low | validation.py `score_holdout_once` | The guard is advisory. It accepts any frame, so a mistaken call with development rows burns the single use and logs it as the holdout evaluation; and nothing prevents scoring `data.X_holdout` directly with `predict` and `rmse_log`, which is not logged. | `score_holdout_once(est, dev.X.iloc[:219], dev.y_log.iloc[:219], "E999")` succeeds | Check that `X_holdout["Id"]` equals the ids in `holdout_ids.json`; keep the final report's rule that only the logged call counts. | Fixed by director (`validation._assert_is_holdout_frame`, called first in `score_holdout_once`, refuses foreign ids, subsets, missing Id, missing file, without consuming the use); verified by reading the code; my xfail marker removed, probe now an ordinary passing test |
| F22 | Info | run_optuna_stage1.py | The overall 3 h tuning cap is recorded in `stage1_summary.json` but not enforced by code (only the 45 min stage cap is); the Stage-2 reference is the lowest `cv_mean` of ANY registered experiment, whatever its feature set or row filter, so the gap is conservative only if all runs share the protocol (currently true: seed 42, same folds). With 10 or fewer trials "improving" compares the first and last running best. | n/a | Add the cumulative elapsed time to the summary and have the director check it before Stage 2. | Note |
| F23 | Info | all three scripts | Every best-of-grid, best-of-trials and best-policy cell is registered with `decision="retain"` automatically. After selection on one set of folds this is provisional. | n/a | Treat as "retain pending finalist check". | Note |

### 7.2 Verified without findings

(a) **Scoring path.** The three scripts contain no reference to the holdout frames, `score_holdout_once`, `load_development_data` or `make_cv`; every grid point, trial and policy cell is scored by `_common.run_cv`, which uses `get_data` (development rows), `make_cv(5, 1, 42)` and the resolved row filter. Probes replace `run_cv` with a spy: the linear grid makes exactly 136 calls (2 x (15 + 13 + 40)), each with `preprocessor_kwargs={"skew_log1p": True}`, a scaler, the requested feature set and filter.

(b) **Optuna kwargs.** For xgboost, lightgbm and catboost, the dict scored by the objective equals the dict stored in the trial's `builder_kwargs`, equals `OPTUNA_SPACES[name](trial)` (plus `extra`, for example `native_categoricals=True`), and the pipeline rebuilt from the JSON round trip has the same repr and the same model `get_params`. For LightGBM, `subsample_freq=1` is in the registered dict and absent from `trial.params`, confirming that rebuilding from `trial.params` would have been wrong. The registry config hash is identical before and after the JSON round trip, so `find_registered` and `evaluate` see the same configuration.

(c) **Stage-2 rule.** Constants are gap 0.003, window 10, cap 3.0 h. A flat study 0.0031 above the reference is not recommended, 0.0029 above is; a study whose running best dropped by more than 1e-4 over the last 10 trials is recommended even when far from the reference; a flat study is not "improving". This matches "within 0.003 of the best model or still improving" (cap enforcement: F22).

(d) **Linear grid.** Only the best Ridge, Lasso and ElasticNet are registered (probe: three records for a 136-point grid, thinned in the probe); each record's `builder_kwargs` contains `scaler` and `preprocessor_kwargs={"skew_log1p": True}`, rebuilds through the builder, and its notes mention the ConvergenceWarning count. The full grid goes to `linear_grid_<feature_set>_<filter>.csv`; warnings are counted per grid point (probe: a warning raised in one call is counted once). Floats survive the JSON round trip exactly.

(e) **Outlier policy script.** Policies are applied through the `train_row_filter` argument of `run_cv` (training rows only); reused cells come from the registry by config hash; dry-run registers nothing.

(f) **Registry integrity (read-only check of the 57 records).** Unique ids, no duplicate config hashes, fold-score count equals splits times repeats, `cv_mean` equals the mean of the fold scores, seed 42 everywhere, `holdout_ids.json` equals the deterministic regeneration, and `holdout_usage.log` does not exist (holdout unused so far).

(g) **Holdout guard (correction 3, R4).** First call scores and logs `{timestamp, frozen_experiment_id, n_holdout, rmse_log, forced=false, n_previous_uses=0}`; a second call raises `HoldoutAlreadyUsedError` and logs nothing; `force=True` scores and logs `forced=true, n_previous_uses=1`; a non-finite prediction raises before logging, so a failed call does not consume the single use. The holdout (219 rows) is disjoint from the development Ids. Residual risk is F21.

### 7.3 Selection bias (for the report)

Measured, not assumed. I re-scored a 15-point Ridge alpha grid (robust scaler plus `skew_log1p`, base features) on the selection folds (seed 42) and on five other partitions (seeds 1-5). The selected alpha 10 scored 0.1319 on the selection folds and 0.1340 on average on the five fresh partitions: optimism +0.0021. The seed-to-seed standard deviation of the CV mean at a fixed configuration is 0.0026. The pure selection effect over 15 neighbouring alphas is about zero, because the curve is smooth and flat near its minimum (the minimum of the seed-averaged curve equals the average of per-seed minima); the optimism comes from the particular fold draw, which is also why a model chosen from a 136-point grid can look 0.002 better than it is.

Reading for the report: a registered best score has an optimism of roughly one seed standard deviation, about 0.002 to 0.003 for linear models. Boosters with 25 TPE trials over 8 interacting parameters have a rougher response surface, so their optimism is probably at least as large (not measured; expect 0.002 to 0.005). Consequences:

- Differences between tuned models below about 0.003 are within selection noise and must not be reported as wins.
- Bounds: (1) a finalist check on FRESH folds (F20: a different seed, or repeats 2-3 only) gives an estimate that is unbiased for the chosen configuration; (2) the single holdout gives the final unbiased number but with only 219 rows its sampling error is large (order 0.01 in RMSE), so it can confirm the level and rule out gross overfitting, not rank close models.

### 7.4 Sign-off

Tuning results admissible to the registry: YES. The outlier-policy, linear-grid and Optuna Stage 1 results may be registered and used for Stage-2 decisions, with the scores described as selection-optimistic (7.3).

Finalist step C8 (RepeatedKFold 5x3): UNBLOCKED, F20 verified fixed (finalist seed 7).

## 8. Round 5 - train.py, predict.py, end-to-end test, finalists runner

Date 2026-11-13. Code at commits `8e1e257` and `d2e4ac7` (read only). Probes: `tests/test_audit_round2.py`, round 5 block (all paths and registries redirected to temp directories; the real registry was not written by me). Full suite after fixes: all passed, no xfails (F24, F25, F26 verified fixed).

### 8.1 Findings

| ID | Severity | Location | Finding | Failing example | Suggested fix | Status |
|---|---|---|---|---|---|---|
| F24 | Medium | train.py `fit_registered`, rows="all" | `--rows all` is accepted before the holdout has been scored. The artefact `<id>_all.joblib` has seen every holdout row; if anyone then passes it (or any model refit from it) to `score_holdout_once`, the single unbiased estimate is destroyed, and nothing in the artefact or the guard says so. The existing end-to-end test itself fits `rows="all"` without a prior holdout use. | `fit_registered("E901", "all", ...)` with an empty `holdout_usage()` succeeds | Refuse `rows="all"` unless `validation.holdout_usage()` is non-empty (or an explicit `force_all` flag that is written to the sidecar). Add `rows_include_holdout: true` to the sidecar and make `score_holdout_once` accept only estimators whose sidecar says `development`. | Fixed by director; verified 2026-11-13 (xfail marker removed, probe passes) |
| F25 | Low | predict.py `predict_test` | Docstring promises "finite and positive after expm1", but only the LOG predictions are checked finite and the price check is `> 0`, which `inf` passes. A log prediction above about 709 would write `inf` into the submission file. Unlikely with real models, reachable with a broken one. | constant log prediction 800 writes `inf` prices (`test_predict_rejects_overflowing_prices`) | `if not np.isfinite(price).all(): raise`. | Open (strict xfail) |
| F26 | Low | run_finalists.py duplicate check | The logic is wrong. `find_registered` builds its probe with `seed=config.SEED` (42), so its config hash can only match a seed-42 record; a finalist registered with seed 7 has a different hash and is never found. The follow-up test "hit has seed == args.seed" can therefore never be true, the reuse branch is dead code, and re-running the script registers a second finalist record with the same configuration (new id, same hash). Scores stay correct; the registry gets duplicates. | run `main(["--ids","E901","--repeats","2","--seed","7"])` twice: two finalist records | Give `find_registered` a `seed` parameter, or scan the registry for equal (builder, kwargs, feature_set, filter, repeats, seed). | Fixed by director; verified 2026-11-13 (xfail marker removed, probe passes) |
| F27 | Info | registry.py `append` | A single `write` of one JSON line, then `write_summary`. A failed finalist run (fit error, interrupt) raises before `append`, so no half-written line is created by the runner (probe). Residual risks: a process killed during the write could leave a truncated last line, which makes `load_registry` raise for everybody; an exception in `write_summary` after the write leaves the line present and the summary stale; there is no file lock, so two writers can take the same `next_id` (this already happened once in practice when two agents registered concurrently). | n/a | Keep the registry lock policy; optionally write to a temp file and append atomically, and let `load_registry` skip a truncated final line. | Note |
| F28 | Info | train.py / predict.py | Artefacts are pickles (joblib): load only your own files. The sidecar records the git commit and library versions (good for R6). `predict` does not warn that a `rows="all"` artefact must not be used for CV or holdout claims. | n/a | Print the `rows` value in the predict report (already present as `rows_fit`). | Note |

### 8.2 Answers to the director's questions

1. **rows="all" before the holdout is scored:** yes it can be invoked, and it should be refused until `holdout_usage()` is non-empty (F24). Training on all rows does not itself leak into CV, but it produces a model that cannot be used for the holdout estimate, and the file name alone does not prevent a mix-up.
2. **Holdout frames in train/predict:** no. Code-level scan (docstrings excluded) finds no `X_holdout`, `y_holdout_log`, `score_holdout_once` or `holdout_usage` in either module. For `development` the fit rows are disjoint from the holdout ids (1,239 rows with the outlier filter, 1,241 without); for `all` they include them. `predict` loads data with the holdout split switched off, uses only `X_test`, and leaves the usage log untouched (probe).
3. **Finalist records vs `evaluate`:** consistent. The record stores builder, kwargs, feature set, filter, `cv_n_splits`, `cv_n_repeats` and `seed` (7); `evaluate.run_registered(<finalist id>)` rebuilds the same pipeline, uses the same repeats and seed, and `reproduces_registered_scores` is True (`test_finalist_registration_reproduces_through_evaluate`, 2 repeats, outlier filter). The runner refuses seed 42 and writes nothing in that case. The comparison CSV is written to a single timestamped file.
4. **Half-written registry line:** not from a failed run; see F27 for the residual cases.

### 8.3 Verified without findings

- `merge_override` merges exactly one level (nested dicts updated, deeper dicts replaced), does not mutate its input, and the override is stored in the sidecar next to the merged `builder_kwargs`; the registry file is byte-identical after a `train` run.
- The registered `train_row_filter` is applied to the fit rows (rows="all" with the outlier filter fits 1,458 rows, without Ids 524 and 1299).
- The sidecar states it is not CV evidence and points to `evaluate`; it records config hash, rows, row counts, commit and versions.
- `predict`: Ids merged onto `sample_submission` (order and length enforced, a missing or duplicated Id raises), log predictions checked finite, price positive, the D18 anomaly Ids are printed.
- `run_finalists`: scores with `validation.cross_validate` on development data only, seed 7 default and 5x3, registers agent "director", tag finalist, new experiment per id with source id in notes, paired difference against the best on the same 15 folds.

### 8.4 Sign-off

C8 finalists tooling (`run_finalists.py`): approved. Do not re-run it for an id that already has a finalist record until F26 is fixed, to avoid duplicate registry lines.

F1 tooling (`train.py`, `predict.py`): approved for `--rows development` and for `predict`. `--rows all` is BLOCKED by F24 until the guard exists; until then run it only after `score_holdout_once` has been called and logged.


## 9. Round 6 - Stage 2, final ensembles, finalists

Date 2026-11-17. Code at HEAD `c9168a0` plus the uncommitted registry growth caused by the running finalist job (the single registry writer). I ran only read-only checks and `python -m houseprices.evaluate` for E066, E067 and E068; nothing else was written except this section. Helper scripts live outside the repository (scratchpad). The finalist records (E076 onward) were NOT inspected before the decision rule in 9.3 was written into this file (the registry had 75 lines when the rule was fixed; it had 77 when the rule was saved, and I looked only at the line count).

### 9.1 Stage 2 procedure (E066, E067)

| Check | Result |
|---|---|
| Registered params equal the best trial of the 75-trial study | E066: trial 43, value 0.1180723, `builder_kwargs` equal to the registry dict, fold scores equal, mean 0.118072. E067: trial 35, value 0.1217333, kwargs and folds equal, mean 0.121733 (includes `subsample_freq=1`, which the space fixes outside `trial.params`). |
| Objective path | `make_objective` calls `_common.run_cv` only, which uses `get_data` (development rows), `make_cv(5, 1, 42)` and the row filter. The script contains no reference to `score_holdout_once`, `X_holdout`, `y_holdout_log` or `holdout_usage` (grep over `experiments/` and `src/`, excluding validation.py itself and docstrings). |
| Resume integrity | SQLite study copy (read-only copy, original untouched): per study 75 COMPLETE trials, numbers 0..74, no gaps, no duplicates, no FAIL/RUNNING rows. The first 25 trials of the live database equal the pre-Stage-2 copy exactly (max difference 0.0, 80 rows compared over four studies). Trial values equal the mean of their five fold scores to 4e-7. |
| Trial CSVs | `stage1_catboost_*_trials.csv` and `stage1_lightgbm_*_trials.csv`: 75 rows each, numbers 0..74, kwargs keys identical in all rows, seed-42 folds. The files are overwritten by each stage (the Stage 1 CSV no longer exists as such; its 25 rows are the first 25 rows). |
| Tuning time | Trial seconds: CatBoost trials 25..74 sum 29.9 min (log: 30.0), LightGBM 3.1 min (log: 3.2). Stage 1: xgboost 2.2 + lightgbm 1.7 + catboost 6.9 = 10.8, native 28.0 (27.97), so 38.8; Stage 2 30.0 + 3.2 = 33.2 (summary `stage_elapsed_minutes` 33.19). Total 72.0 of 180 min. The claim is correct. |
| Registered record content | Title, tags `["tuning","stage2",...]` and top-3 notes are right; interpretation figures (median 0.1201 and worst 0.1298 for CatBoost; 0.1256 and 0.1593 for LightGBM) equal the trial CSVs. The statements in `EXPERIMENT_HISTORY.md` that 58 of 75 CatBoost trials lie within 0.003 of the best (computed: 58), that the best-so-far curve is flat over the last 10 trials (computed: yes, both boosters), and that the trial SD fell from 0.0028 to 0.0005 (computed 0.0027 to 0.0005) and 0.0069 to 0.0016 are confirmed; the 0.0028 is a rounding difference. |
| Search box | The registered CatBoost has `iterations=2000`, the upper edge of the space, and LightGBM `n_estimators=2000` and `min_child_samples=5`, edges of their spaces; the limitation recorded in D21 is correct and means the tuned optimum is a boundary solution, not a stationary point. |

### 9.2 Registry, reproducibility, ensembles, holdout

- **Registry integrity (E066-E075).** 75 parseable lines at the time of the check, unique ids, no duplicate config hashes (hash recomputed through `Experiment.config_hash`), `cv_mean` equals the mean of the five fold scores for all records (maximum difference 6e-7, rounding), `cv_std` equals the sample SD (ddof 1), seed 42, k 5, repeats 1, feature set equal to the working set `base+areas+baths+ages+quality_interactions+ordinal_quality`, filter `keep_all`, builders resolvable. Titles and tags of E066/E067 say stage 2; E068-E075 carry `ensemble`, kind and group tags. Decisions: `retain` (E066, E067), `reject` (E074), `investigate` (the rest), consistent with the script's one-SE rule.
- **Reproduction.** `python -m houseprices.evaluate --experiment E066`: `reproduces_registered_scores` true (0.118070 vs 0.118072, five fold scores equal to 5 decimals). E067: true (0.121730 vs 0.121733). E068: true. Nested ensembles E069-E075 were deliberately not re-run (CPU shared with the finalist job); see 9.5 for the one reproduced after the run.
- **Ensemble leakage.** Re-read `run_ensembles.py` and `ensemble.py`: every ensemble is scored through `_common.run_cv` (outer 5-fold, seed 42, development rows). `BlendingRegressor.fit` learns NNLS-type weights from `inner_oof_matrix` of the rows it receives; `NestedStackingRegressor.fit` fits the Ridge meta-learner on that inner OOF matrix; the bases are refit on the received rows; the average has fixed weights 1/k and learns nothing. The diversity pass (`run_cv` per base, then `ensemble.diversity_report` and CSV) feeds no fit and no registered number; its only other effect is on the paired table in dry-run mode (`b.folds`, `b.cv_mean` replaced only when `--dry-run`). The registered bases' cv_mean were recomputed in the diversity pass and match the registry to 1e-6 (log lines in the history). Pooled OOF RMSEs in the history (CatBoost 0.1190, LightGBM 0.1228, XGBoost 0.1242, Lasso 0.1272) equal `diversity_20261116_234858.csv`.
- **Paired arithmetic.** I recomputed delta and paired SE for all eight rows of `ensembles_20261116_234858.csv` from the registry fold scores (ensemble minus its best member, same five folds, ddof 1, sd/sqrt(5)); all differ from the CSV by less than 2e-6 (E068: -0.000728, SE 0.001943; E072: -0.000594, SE 0.001650; E071: +0.000650, SE 0.000943). Fold-level differences E068 minus E066: -0.0018, +0.0057, -0.0023, -0.0061, +0.0009; E072 minus E066: -0.0024, +0.0056, -0.0021, -0.0039, -0.0001. The ensemble advantage is carried by folds 1, 3, 4 and reversed in fold 2, which is the fold with the largest errors in every model.
- **Stage 2 versus Stage 1 (paired, same folds):** E066 minus E060 = -0.00139 (SE 0.00065; better in all five folds); E067 minus E059 = -0.00268 (SE 0.00084). Both are selection-conditional (best of 75 versus best of 25 on the folds used to select), so they are upper bounds on the real improvement.
- **Holdout.** `experiments/holdout_usage.log` does not exist; `holdout_ids.json` has 219 ids (seed 42, fraction 0.15, 16 strata), a single commit in its history (`582bf86`) and is not modified in the working tree. SHA-256 of `HousePricing.ipynb` (1BB32B57...), `train.csv` (1E18ADDF...) and `test.csv` (8FDD3D82...) equal the manifest in `docs/AUDIT_2026-11-11.md`. `git status` shows only the director's uncommitted history/registry edits and `outputs/`.

### 9.3 Selection-optimism accounting and the a-priori decision rule

Fixed before any finalist record was read. Evidence used: audit 7.3 (a fixed configuration moves with the fold draw by one seed SD, 0.0026; the Ridge optimism measured +0.0021; Lasso E056 on fresh seed-7 folds 0.1282 versus 0.1255, +0.0027), the Stage 2 trial tables, and the ensemble fold scores.

**(a) Best-of-75 CatBoost E066 on the seed-42 folds.** Two components. (i) Fold-draw component: the seed-42 partition is the one every configuration was ranked on; a fresh partition moves the level of any model by about one seed SD (0.0026) in either direction on average, and measured optimism for the tuned linear models is +0.002 to +0.003. This component is almost entirely COMMON to all models (it cancels in paired differences), so it matters for the absolute level and not for the ranking. (ii) Pure winner's curse among 75 TPE trials: bounded above by the gap between the best trial and the mean of the top ten (CatBoost 0.0010; LightGBM 0.0007), because the top ten contain real differences as well as noise; the trial SD in the last 25 trials is only 0.0005. I therefore expect E066's seed-7 5x3 mean to lie at about 0.1180 + 0.0020 (fold draw) + 0.0005 to 0.0010 (winner's curse) = **0.1205 to 0.1220**, central 0.1210, and I would call a seed-7 value below 0.1195 or above 0.1235 surprising and worth an explanation before a freeze. For ranking between tuned CatBoost and E060 (best of 25), the differential optimism is about 0.0005 to 0.0008 (more trials, more curse), so the observed -0.0014 paired gain is expected to shrink to about -0.0006 to -0.0009 on fresh folds.

**(b) "Best of 8 ensembles" on the same folds.** The eight ensembles are strongly correlated; the six non-linear-only ones span paired differences against E066 from -0.0007 to +0.0009 (SD 0.0006) and the effective number of independent choices is about three, so the expected gain from choosing the minimum is about 0.85 x 0.0006 = **0.0005 (range 0.0003 to 0.0008)**. A second, partly offsetting effect: the ensemble carries only half of CatBoost's tuning luck on these folds (and half of Lasso's, selected from a 136-point grid), so its level optimism is slightly smaller than CatBoost's alone, which moves the fresh-fold advantage of the average a little in its favour (about +0.0002 to +0.0005 expected shift). Net expectation: the true advantage of the best two-member ensemble over E066 is about -0.0005 (that is, E068's observed -0.0007 is explained to roughly 70% by selection), with sampling noise on a seed-7 5x3 mean difference of about 0.0012 (paired SD of fold differences 0.0043 per fold, 5 effectively independent folds per repeat, 3 repeats that overlap in rows). The six-base and three-booster ensembles and the blends have no expected advantage.

**Why the runner's SE is not used (F30).** `run_finalists.py` computes the paired SE as sd/sqrt(15). The 15 folds of a `RepeatedKFold(5, 3)` are three re-partitions of the same 1,241 rows, so they are not independent; effective n lies between 5 and 15 and the printed SE and the column `within_one_se_of_best` are understated by up to a factor 1.7. I use SE_adj = sd of the 15 paired differences / sqrt(5).

**A-priori rule (Agent D, committed to the file before the finalist table was read).** Candidates are compared on the seed-7 5x3 registry records (fold scores recomputed by me from `registry.jsonl`, not from the runner CSV). Let d = mean over the 15 folds of (candidate minus E066), SE_adj as above, and r1..r3 the three repeat means of the differences.

1. Default = single CatBoost E066 (simplicity, one model, one fit, no learned second layer).
2. An ensemble X from {E068, E072, E069} replaces E066 only if ALL hold: (i) d <= -0.0010 (margin = expected selection inflation 0.0005 plus half of it again as allowance for the offsetting effect and rounding); (ii) d <= -1.0 x SE_adj (about -0.0019 at the fold SD seen on seed 42); (iii) the sign is negative in all three repeat means r1..r3 and in at least 10 of the 15 folds. If only (iii) holds, or d is between -0.0010 and 0, the result is a tie and E066 stays by parsimony. If d > 0 for every ensemble, E066 stays and the ensemble idea is closed.
3. Six-base, three-booster and linear ensembles (E070, E071, E073, E074, E075) are not eligible unless their d is more negative than the best of rule 2's candidates by more than 0.0010 and they satisfy (i)-(iii) themselves (not expected).
4. E066 versus E060: keep E066 unless E060's d (E060 minus E066) is <= -0.0005 and <= -1.0 x SE_adj, in which case the Stage 2 gain is not real on fresh folds and E060 (cheaper, fewer iterations) is preferred. E067 (LightGBM) and E056 (Lasso) are reference points only; neither can be frozen over E066 unless their d is negative with the same (i)-(iii) bar.
5. Absolute sanity bounds: freeze is blocked if the seed-7 mean of the chosen model is below 0.1185 (implausibly optimistic for a fresh partition; look for leakage) or above 0.1250 (more than twice the expected optimism; look for fold-set sensitivity), or if the chosen model's worst repeat mean exceeds its best by more than 0.0040.
6. The holdout (219 rows, sampling error of the order 0.01) is not used for ranking; the frozen id is the one chosen by rules 1-5.

I deliberately set the bar above the director's standing 0.003 band for single-fold-set comparisons (D14) in the sense that it is paired and has a sign-consistency condition, and below it numerically because paired differences between near-identical models are much less noisy than unpaired levels (SE_adj about 0.002 versus a level SD of 0.0026 per partition). If the director prefers the strict 0.003 band, the outcome can only be "E066 stays".

### 9.4 Findings

| ID | Severity | Location | Finding | Evidence | Suggested fix | Status |
|---|---|---|---|---|---|---|
| F29 | Low (documentation) | `run_optuna_stage1.py` `register_best` texts | E066/E067 carry hypothesis and rationale generated for Stage 1: "a small seeded TPE search ... before deciding on a larger Stage 2", and `implementation` names the study `stage1_catboost_...`. Title (`Optuna stage 2: ... best of 75 trials`) and tag `stage2` are right; the study name is shared across stages by design. A reader of the registry alone could believe the record is a Stage 1 result. | E066 and E067 JSON lines | The registry is append-only: leave the records, state in the report that E066/E067 are resumed studies (25 + 50 trials; first 25 trials identical to E060/E059's study), and, when the script is next touched, make `rationale` depend on `--stage`. | Open (note) |
| F30 | Medium (inference, finalist step) | `run_finalists.py` lines 111-114 | Paired SE uses sd/sqrt(15) and the flag `within_one_se_of_best` is derived from it. The 15 folds are 3 repeats over the same rows, so the folds are positively dependent and the SE is understated by up to 1.7 times, so the "within one SE" test is too strict (more candidates look distinguishable from the best) and any small advantage of a challenger looks more significant than it is. | Design of `RepeatedKFold(5,3)`; the same 1,241 rows are scored three times | Report SE_adj = sd/sqrt(5) next to the runner's value (done by Agent D in 9.6); do not change the runner during the finalist run. | Open (handled in my table) |
| F31 | Info | `stage1_summary.json`, trial CSVs | Both files are overwritten per stage: after Stage 2 the summary has the 75-trial statistics for catboost/lightgbm; the Stage 1 summary is recoverable only from the trial CSV (first 25 rows, intact) and from the gitignored `optuna_stage1.pre_stage2.db` (checked: identical first-25 values). The sampler is re-created with `TPESampler(seed=42)` on resume, so the 50 resumed trials are not the draws of an uninterrupted 75-trial seed-42 run; this does not affect validity (no information from a fold other than the training folds is used) but the study is not exactly re-creatable. | DB comparison above | Mention in the report; keep the `.pre_stage2.db` copy outside git ignore if the study must be archived. | Note |
| F32 | Info | `run_ensembles.py select_best` | The bases are chosen by lowest registered `cv_mean` among all single models of a family on the same seed-42 folds (E056 from a 136-point grid, E066 from 75 trials). Weights and meta-learner are nested, base hyper-parameters are not. This is the known selection optimism of 9.3, shared by every ensemble member; it is not leakage of the outer fold into weights. | `select_best` and the history table | State in the report that ensemble scores on seed 42 are selection-optimistic in the same way as their bases and are decided on seed 7. | Note |

### 9.5 Verified without findings

- Seed, k, repeats, feature set and filter of E066-E075; no duplicate ids or hashes; cv_mean and cv_std arithmetic.
- Scoring paths of the tuning script and of the ensemble runner (outer CV only); no holdout symbol in either; the interpretability runner (`run_interpretability.py`) only names `score_holdout_once` in its docstring (it says it never calls it).
- Fold-score and kwargs equality between best trial and registered record; no trial-number duplication after resume; first 25 trials unchanged by Stage 2.
- Tuning-time arithmetic 10.8 + 28.0 + 30.0 + 3.2 = 72.0 minutes.
- Reproduction of E066, E067, E068 through `evaluate` (true).

### 9.6 Finalist table (seed 7, RepeatedKFold 5x3) and application of the rule

Run finished (registry 83 lines, `experiments/finalists/finalists_20261117_003230.csv` written, no experiment process left). Verified: 8 finalist records E076-E083, one per source (E066, E068, E072, E069, E060, E061, E056, E067), source id in `notes`, agent `director`, tags start with `finalist`, seed 7, k 5, repeats 3, 15 fold scores each, `cv_mean` equals the mean and `cv_std` the sample SD, builder, kwargs, feature set and filter equal to the source record, no duplicate ids, no duplicate config hashes (seed 7 gives new hashes; F26 stays fixed). The runner's CSV agrees with the registry (fresh mean, optimism, paired difference against its reference E077). Reproduction after the run: `evaluate --experiment E076` (RepeatedKFold 5x3 seed 7) true, 0.12166 vs 0.121664; `evaluate --experiment E072` (nested stack, 5-fold seed 42) true, 0.11748 vs 0.117479.

I recomputed the table from the registry fold scores. d = mean paired difference to E066 (negative = better than E066), SE_adj = sd of the 15 differences / sqrt(5), r = the three repeat means of the difference.

| source | finalist | seed-7 mean | seed-42 mean | shift | d vs E066 | SE (runner, /sqrt15) | SE_adj | d/SE_adj | repeat means of d | folds better than E066 | repeats better |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E068 average Lasso+CatBoost | E077 | 0.12066 | 0.11734 | +0.0033 | -0.00101 | 0.00176 | 0.00305 | -0.33 | -0.00074, -0.00102, -0.00126 | 9 of 15 | 3 of 3 |
| E061 average, Stage 1 CatBoost | E081 | 0.12147 | 0.11807 | +0.0034 | -0.00020 | 0.00210 | 0.00365 | -0.05 | +0.0001, -0.0002, -0.0005 | 9 | 2 |
| E072 stack Lasso+CatBoost | E078 | 0.12164 | 0.11748 | +0.0042 | -0.00002 | 0.00160 | 0.00278 | -0.01 | -0.0002, +0.0006, -0.0005 | 10 | 2 |
| **E066 CatBoost Stage 2** | **E076** | **0.12166** | 0.11807 | +0.0036 | 0 | | | | | | |
| E069 blend Lasso+CatBoost | E079 | 0.12240 | 0.11802 | +0.0044 | +0.00074 | 0.00151 | 0.00261 | +0.28 | +0.0007, +0.0011, +0.0004 | 9 | 0 |
| E060 CatBoost Stage 1 | E080 | 0.12325 | 0.11946 | +0.0038 | +0.00158 | 0.00064 | 0.00112 | +1.42 | +0.0017, +0.0015, +0.0016 | 5 (E060 better) | 0 |
| E067 LightGBM Stage 2 | E083 | 0.12524 | 0.12173 | +0.0035 | +0.00357 | 0.00094 | 0.00162 | +2.20 | +0.0044, +0.0028, +0.0035 | 4 | 0 |
| E056 Lasso | E082 | 0.12816 | 0.12545 | +0.0027 | +0.00650 | 0.00380 | 0.00659 | +0.99 | +0.0067, +0.0067, +0.0061 | 6 | 0 |

(The fold column counts folds in which the challenger beats E066; E069 wins 9 of 15 folds but loses by 0.012 in one fold per repeat, so its mean is worse. For E060, E067 and E056 the challenger wins 5, 4 and 6 folds, so E066 wins 10, 11 and 9. The last column counts repeats in which the challenger's mean beats E066.)

Every model moved up by 0.0027 to 0.0044 on fresh folds, the common fold-draw effect predicted in 9.3 (one seed SD, 0.0026). E066's seed-7 mean 0.12166 lies inside my predicted band 0.1205-0.1220 (central 0.1210), and E066's repeat means (0.12021, 0.12263, 0.12215) have a range of 0.0024, below the 0.0040 limit.

**Rule outcome.**

- E068 (best ensemble): d = -0.00101 passes (i) by 0.00001 only, fails (ii) (needs d <= -0.0031; d/SE_adj = -0.33) and the 10-of-15 part of (iii) (9 of 15), although all three repeat means are negative. Tie; **E066 stays by parsimony**.
- E072: d = -0.00002; E061: -0.0002; E069: +0.0007. None passes (i). Closed.
- E066 versus E060: E060 is worse by 0.00158 (SE_adj 0.0011), in all three repeats, and E066 wins 10 of 15 folds. The Stage 2 gain survives fresh folds (seed 42: -0.00139, seed 7: -0.00158); rule 4 keeps E066. The gain is about 0.0015, below the 0.003 noise band, so it should be described as a consistent small gain, not a decisive one.
- E067 and E056 are clearly behind E066 (d/SE_adj +2.2 and +1.0; in absolute terms 0.0036 and 0.0065).
- Sanity bounds (rule 5) hold for E066: seed-7 mean 0.1217 is between 0.1185 and 0.1250; repeat range 0.0024.

**Two honest caveats for the report.** (1) The point estimate favours the average E068 by 0.0010 on seed 7 (and by 0.0007 on seed 42; the two agree, which is what the 9.3 expectation of a true advantage of about 0.0005 predicts). The evidence is a tie, not a defeat: the uncertainty on d is about 0.003, so the data cannot exclude an ensemble advantage of 0.002 or a disadvantage of 0.002. Choosing E066 is a parsimony decision (one model and one fit, no second base to maintain, no clip placement inside two base specs; the average costs about the same to fit, 97 s versus 84 s, the stack 490 s), not a finding that the ensemble is worse. (2) In both seeds, the paired differences are dominated by one fold per repeat in which the Lasso partner hurts by about 0.010 (seed 7: fold 4 in repeats 1 and 2, fold 5 in repeat 3, +0.008 to +0.011; seed 42: fold 2, +0.006), and by one or two folds where it helps by 0.004 to 0.010. The ensemble is therefore a bet on a handful of houses, which is the same reading as the residual-correlation table (linear-tree 0.83-0.86) and the reason the SE is large.

### 9.7 Findings added after the finalist run

| ID | Severity | Location | Finding | Evidence | Status |
|---|---|---|---|---|---|
| F30 (update) | Medium | `run_finalists.py` | Quantified: the three repeat means of every paired difference agree within 0.0003 to 0.0008 (E068: -0.00074, -0.00102, -0.00126), while the fold-level SD of the differences is 0.006. Repeats add almost no information on the difference beyond averaging the partition noise, so "all three repeats agree in sign" is weak evidence, and the real uncertainty is sampling of the 1,241 rows, about sd/sqrt(5) = 0.003 for the ensemble comparisons. The runner's column `within_one_se_of_best` (SE 0.0016-0.0018) is anti-conservative for any claim of a significant gap and too strict for a "within SE" tie. | `r6_final.py` output above | Open: use SE_adj in the report; the registry is unaffected |
| F33 | Info | finalist records | `E080` is tagged `stage1` and `E083`, `E076` `stage2`; the finalist title repeats the source title ("Optuna stage 2: ..."), so the title alone does not say "finalist, seed 7"; the tag and `notes` do. | registry | Note |

### 9.8 Sign-off

- **Stage 2 results admissible: YES.** E066 and E067 are the best of their 75-trial studies, registered with exactly those parameters, scored only through the development-row CV, reproduced through `evaluate`, with the 72 of 180 minute tuning claim verified. Describe them as selection-optimistic: on fresh folds the level rises by 0.0036 (E066) and 0.0035 (E067), and the Stage 2 gain over Stage 1 is about 0.0015 for CatBoost (seed 42 -0.0014, seed 7 -0.0016) and 0.0027 for LightGBM (seed 42 only; E067 is behind E066 by 0.0036 on seed 7, so it does not matter for the final choice). Limitations to state: parameters on the edge of the search box (2,000 iterations, `min_child_samples` 5), resumed sampler (F31), stale rationale text (F29).
- **Ensemble results admissible: YES.** No leakage of the outer fold into weights or meta-learner; paired arithmetic reproduced to 2e-6 for all eight rows; E068 and E072 reproduced; the diversity pass is diagnostic only. The seed-42 scores are selection-optimistic like their bases (F32). No ensemble clears the one-SE bar on seed 42 and none passes the a-priori rule on seed 7.
- **Finalist table admissible: YES**, with F30: report SE_adj, not the runner's SE. Records are correct and unduplicated, seed 7 shares no fold with seed 42 (F20), reproduction true.
- **Blocks for a freeze: none from validation.** Holdout never scored (`holdout_usage.log` absent), 219 ids unchanged (single commit), originals' SHA-256 equal to the manifest. Conditions for the freeze step: (1) freeze E066 (finalist record E076) under rule 2, record the decision text including the E068 caveat; (2) `train --experiment E066` on development rows only, with the year clipper (`clip_years`) as an override only if the director wants it, which changes the model relative to E066's registered CV and must then be treated as a different configuration (unregistered CV evidence) and justified separately; (3) exactly one call to `validation.score_holdout_once`, then `--rows all` is unlocked (F24); (4) the 219-row holdout confirms the level (expect about 0.12 to 0.13 plus its sampling error of order 0.01) and cannot rank E066 against E068.

### 9.9 Freeze approval (D22, commit 23f56ac) and pre-registered holdout range

Checks, 2026-11-17, before any holdout use: (a) D22 equals my rule 2 and 4 outcome (E066 stays, E068 a tie at d = -0.0010, E066 over E060); its builder kwargs equal registry E066 exactly and the recomputed config hash is `be58905cb40a`; no override, no `clip_years` (so `evaluate --experiment E066`, reproduced true, is the exact evidence). (b) `experiments/holdout_usage.log` absent; `holdout_ids.json` unchanged (single commit `582bf86`); registry still 83 lines; `outputs/models` does not exist yet. (c) HEAD is `23f56ac`; `git status` shows only `outputs/` and one untracked file, `experiments/finalists/score_holdout.py`; no tracked file modified; suite 215 passed. I read `score_holdout.py`: it checks the sidecar (experiment id, config hash equal to the registry, rows = development, `includes_holdout_rows` false, no override), loads the development data, makes ONE `score_holdout_once` call on the registered holdout frame, then re-predicts the same model on the same rows to save residuals (deterministic repeat of the logged call; the assertion could abort after logging only if CatBoost is non-deterministic beyond 1e-6, which loses the CSV but not the score; if so, do not predict again, report the logged value). Commit the script before running so the tree is clean (the script is not pipeline code).

Pre-registered expectation for the 219-row holdout RMSE (log scale) of E066 refit on the 1,241 development rows. Basis: seed-7 CV mean 0.1217 (fold SD 0.0131, 15 folds of about 248 rows range 0.1027 to 0.1501, 10th-90th percentile 0.109 to 0.140); the final fit uses 100% instead of 80% of the development rows (worth about -0.001); holdout sampling error for 219 rows about 0.0138 (SD) and right-skewed because a few large-error houses dominate. Expected centre **0.121**. **Consistent with the CV estimate: 0.100 to 0.145.** Below 0.100 is better than anything seen in 15 folds and would be luck, not a finding; above 0.145 up to 0.150 is a bad draw within the fold range; **above 0.150 or below 0.095 I will report as inconsistent and look for a data or pipeline fault (never for a tuning remedy)**. The comparison is level only; the holdout cannot rank E066 against E068 (sampling error 0.014 against a 0.001 difference).

**APPROVED: freeze E066 (D22) and the planned holdout procedure, subject to committing `score_holdout.py` first.**


## 10. Final round

Date 2026-11-17, repository at `385a542`. Read-only on pipeline code. I did not call `score_holdout_once`, did not predict on the holdout, did not run `train`/`predict` into the real `outputs/`, and did not write the registry. The independent rebuild ran in memory (no file written). Added tests: two in `tests/test_validation.py` (final holdout log state and submission validity). Probes and scripts are outside the repository.

### 10.1 Holdout discipline

- `experiments/holdout_usage.log` has exactly one line: 2026-11-17 00:40:45, E066, 219 rows, RMSE 0.10645, `forced` false, `n_previous_uses` 0. `holdout_ids.json` still has one commit (`582bf86`) and 219 ids.
- Sidecars: `E066_development.json` rows development, 1,241 rows, `includes_holdout_rows` false, `holdout_scored_before_fit` false, hash `be58905cb40a`, fitted 00:40:42, three seconds BEFORE the logged scoring; commit field `264f43f` (the script commit; the pipeline commit is `23f56ac`). `E066_all.json` rows all, 1,460 rows, `holdout_scored_before_fit` true, 00:41:04, after the log line. `git diff --stat 23f56ac HEAD -- src` is empty and `git log 23f56ac..HEAD` touches only the scoring script, the predictions CSV, the usage log, the submission and docs; no change to `src/`, to the registry, or to tuning/ensemble code after the freeze. The two sidecars carry identical builder kwargs equal to registry E066.
- `holdout_predictions_E066.csv`: 219 rows, Id set equals `holdout_ids.json`, no duplicates, `residual_log` equals `y_log - pred_log` (3.6e-15), RMSE recomputed 0.106450, equal to the logged value.

### 10.2 Why the holdout is below the CV estimate, and what may be claimed

Numbers (my scripts, E066 only): 
- Seed-7 fold scores range from 0.1027 to 0.1501 (mean 0.1217, SD 0.0131); 0.1065 lies inside that range, close to its low end (it equals the 2nd best of 15 folds).
- The four houses above 4,000 sq ft of living area (Ids 524, 692, 1183, 1299) are all development rows; the holdout's largest GrLivArea is 3,627. In the seed-42 out-of-fold predictions of E066 those four houses have log errors +0.669, -0.529, -0.636, +0.603 and carry 8.5% of the total squared error. Pooled OOF RMSE: 0.1190 with them, 0.1140 without (-0.0050). By fold: fold 2 goes from 0.1323 to 0.1159 and fold 3 from 0.1391 to 0.1327 when they are removed; the other three folds contain none of them (0.1039, 0.1023, 0.1128).
- The holdout's own errors: mean residual +0.004 (no bias), MAE 0.076, largest |error| 0.41 (Id 711); the five worst houses carry 24% of the squared error, and without the worst three the RMSE would be 0.0977. Holdout log price mean 12.028, SD 0.403, against development 12.023 and 0.399, so the sample is not unusual in price level.
- The development-row fit used 100% of 1,241 rows against 80% in the folds (worth roughly 0.001).

Reading: three effects point the same way, in this order of size: no extreme-GrLivArea houses in the holdout (about 0.005), the fold-draw and sampling noise of 219 houses, and the larger training set (about 0.001). The score is not evidence of leakage: the model was fitted before scoring, the holdout was untouched, and nothing in the procedure was tuned to it.

Interval (bootstrap of the 219 squared residuals, 20,000 resamples, seed 0): RMSE 0.1065, **95% interval 0.0922 to 0.1206**, bootstrap SD 0.0073. The pre-registered consistent range (0.100 to 0.145, centre 0.121) is met. Note that the seed-7 CV estimate 0.1217 lies just above the upper end of this interval (0.1206); with the extreme houses removed from the comparison the gap disappears (0.1140 seed-42 pooled OOF without them, plus the 0.003 to 0.004 fresh-fold shift, about 0.117 to 0.118).

**May it be reported as an unbiased estimate?** Yes, as an unbiased single draw of the generalisation RMSE of the development-row fit of E066 on houses like the holdout, with the stated uncertainty (about 0.007 sampling SD, interval 0.092 to 0.121, which ignores the variance of the model fit). Say all of the following with it: (1) it is one 219-house sample with no house above 4,000 sq ft, so it is probably favourable relative to the competition's test set; (2) the more conservative level is the seed-7 CV estimate 0.1217; (3) the submitted model is the refit on all 1,460 rows, which has seen the holdout, and has no separate error estimate; (4) the holdout was not used to choose between E066 and the ensembles and cannot (sampling error about 0.014 against a 0.001 difference).

### 10.3 Submission file and independent rebuild

- `outputs/submissions/E066_all.csv`: columns `Id, SalePrice`; 1,459 rows; Id order equal to `sample_submission.csv`; no duplicates; all prices finite and positive. Prices: min 41,901, median 156,611, mean 177,572, max 524,842 (train: 34,900; 163,000; 180,921; 755,000). Log-price mean 12.009, SD 0.390 (train 12.024, 0.399); quantiles 1/5/25/50/75/95/99%: 66.7k, 91.2k, 127.6k, 156.6k, 208.2k, 335.4k, 435.3k (train 61.8k, 88.0k, 130.0k, 163.0k, 214.0k, 326.1k, 442.6k). Spearman of the prediction with OverallQual 0.84 and with GrLivArea 0.74. Lowest predictions are 729-936 sq ft quality-3 houses (Ids 1823, 2892, 2894), highest are 2,800-3,500 sq ft quality 9-10 houses (Ids 2628, 2629, 2683). Nothing implausible; the test median is 4% below the training median, the usual tree-model shrinkage towards the centre, and the maximum 524,842 is below the training maximum 755,000 because no test house resembles the two largest.
- Anomaly Ids: 1877 -> 212,580; 2550 -> 268,438; 2593 -> 256,431. These are unvalidated (no training house looks like them, residual risk 3 of the claims packet); they are in a plausible range for their descriptions, but I cannot verify them by data.
- **Independent rebuild:** my own lines (registry record E066, `data.load_development_data(feature_set, with_holdout_split=False)`, `fit(X, y_log)` on 1,460 rows, `expm1` of the test predictions, merged on Id). Maximum relative difference to the submission 3.1e-16, mean 2.6e-17: bit-for-bit identical up to float printing. CatBoost is deterministic here (fixed `random_seed` in the builder, fixed thread count).

### 10.4 Registry, originals, suite

- Registry: 83 records, ids unique and sequential E001-E083, no duplicate config hash (the eight finalist records differ from their sources by seed), `cv_mean` equals the mean of the fold scores (maximum difference 6e-7), fold count equals splits times repeats everywhere, seeds 42 (75 records) and 7 (8 records).
- Originals: SHA-256 of `HousePricing.ipynb`, `train.csv`, `test.csv` begin 1BB32B57, 1E18ADDF, 8FDD3D82, equal to the manifest.
- Suite: 216 of 217 tests pass; the one failure is finding F34.

### 10.5 Findings

| ID | Severity | Location | Finding | Evidence | Status |
|---|---|---|---|---|---|
| F34 | Low (test hygiene) | `tests/test_holdout_guard.py::test_real_log_is_still_empty` | The test asserts that the real usage log is empty ("check before freezing"). It is now false by design, so the suite is red (1 failed, 214 passed in my run before my two additions). Not a pipeline fault. | pytest output | Open: the director should replace it (my file `tests/test_validation.py` now contains `test_final_holdout_log_is_one_unforced_E066_entry_and_matches_saved_predictions`, which states the new invariant: one line, E066, unforced, 219 rows, predictions CSV reproduces the logged RMSE) and delete or invert the old one, then the suite is 217 of 217 |
| F35 | Info | `score_holdout.py` | The script predicts on the holdout twice inside one run (the logged call, then a re-prediction to save residuals) and asserts equality to 1e-6. The saved residuals reproduce the logged RMSE to 1e-15, so no second evaluation was made in substance; the report should still say "one scoring call" and not "one prediction". | script and CSV | Note |
| F36 | Info | holdout composition | No house above 4,000 sq ft in the holdout (decile stratification on price did not control area outliers). It lowers the expected holdout RMSE by about 0.005 relative to the development set. | 10.2 | Note: state in the report |

### 10.6 Verified without findings (this round)

Single usage line, forced false; sidecar fields and order of events; no change to `src/`, registry or experiment code after the freeze commit; holdout predictions CSV (ids, arithmetic, RMSE); submission format, order, finiteness and distribution; bit-level rebuild of the submission from the registry record; registry integrity for 83 records; originals unchanged; Stage 2, ensemble and finalist results as in section 9.

### 10.7 Residual risks for the report

1. Selection optimism: seed-42 scores (E066 0.1181) are optimistic by about 0.003 to 0.004 on fresh folds (seed 7: 0.1217); quote the seed-7 figure as the CV estimate.
2. The holdout is one 219-house sample (interval 0.092 to 0.121), has no house above 4,000 sq ft, and cannot rank close models; the E066 versus average-ensemble choice is a parsimony decision on a tie (paired difference -0.0010, adjusted SE 0.003).
3. The submitted model is refit on all rows; its own error is unmeasured.
4. Test-only anomalies (Ids 2593, 2550, 1877, 2121): predictions unvalidated.
5. Tuned parameters sit on the edge of the search box (2,000 iterations; LightGBM `min_child_samples` 5); a wider search could change the tuned level slightly, not the conclusion.
6. Reconstructed cleaning rule C01 (1 training row, 9 test houses), registry writes not atomic (F27), pickled artefacts (F28), pandas/scikit-learn versions pinned (only pandas 2.3.3 run), F30 (runner SE too small), F29/F31/F32 notes.
7. Independence: this audit is by an agent of the same team; the external reviewer's read of the repository is the independent check.
8. No Kaggle leaderboard score exists and none is claimed; the file was not submitted.

### 10.8 Final sign-off

- Holdout discipline: **clean** (one unforced use, after the freeze, model fitted before, nothing changed afterwards).
- Reported holdout score 0.10645 with interval 0.092 to 0.121: **admissible as an unbiased single estimate with the four qualifications of 10.2**.
- Submission `outputs/submissions/E066_all.csv`: **valid and reproducible**; fit for manual upload.
- Registry and originals: **intact**.
- Open items before the final commit: fix the stale guard test (F34); re-count the suite in the claims packet (217); documents must quote numbers as in 10.7 and the claims packet section F.

**Final statement: from the validation side there is nothing that blocks reporting or submitting E066.**

### 10.9 Document consistency pass (2026-11-17; findings resolved, see end of section)

Read-only pass over README, RESEARCH_REPORT, EXECUTIVE_SUMMARY, the packet intro, STUDY_GUIDE parts 0, 4, 6, 7 and the last EXPERIMENT_HISTORY sections. No wrong score, difference, SE, count, price, commit or experiment id was found; no stale "pending" text. Study guide part 5 was only spot-checked. Items reported to the director:

1. Wording stronger than the evidence: README and RESEARCH_REPORT section 1 say the seed-7 folds were "not used for any choice"; they were used once, under the pre-registered rule, to choose among eight finalists.
2. RESEARCH_REPORT section 2: finalist folds "disjoint from the selection partition"; F20 verified only that no fold is identical (validation overlap up to 25%).
3. EXECUTIVE_SUMMARY ("each with a test") and STUDY_GUIDE part 6 section 6 ("each audit finding became a test"): Info/Note findings have no test.
4. RESEARCH_REPORT 4.5: LightGBM minutes 4.8 should be 4.9 (1.7 + 3.2); the 72.0 total needs 4.9.
5. STUDY_GUIDE part 6: "about 2,600 lines, roughly 215 tests" should be 2,439 lines and 217 tests; `--dry-run` is `--dry-run-rows N` in run_outlier_policy.py and run_linear_family.py.
6. STUDY_GUIDE part 4: "eight candidates" but seven rows (E061/E081 missing); "a quarter to a half of one SE" should be 0.26 to 0.57.
7. STUDY_GUIDE part 7: L1 beats Ridge "by 0.005 in every fold"; per-fold gaps are 0.003 to 0.007, mean 0.0048.
8. RESEARCH_REPORT section 1: "70% of the mean baseline's error on the same folds"; baseline is seed 42, final figure seed 7 (69.5%).
9. README/EXECUTIVE_SUMMARY "seven audit rounds": the audit labels eight (1, 2, 2b, 3, 4, 5, 6, final) unless 2b is folded into 2.

Verified correct: holdout figures and interval, all finalist/ensemble tables, error-concentration tables, permutation/SHAP rankings, seed-7 fold composition (0.1139 / 0.1255), submission statistics, hashes and commit ids, part 6 module and guard names.

**Resolution (director, 2026-11-17).** The director's list of record was `docs/PENDING_FIXES.md` (11 items: the nine above, with item 5 split into line/test counts and dry-run flags, and item 6 into the missing row and the SE ratio). Each was re-checked against the registry and saved artefacts before editing, with no experiment re-run and no holdout access:

| item | check | change |
|---|---|---|
| 1 | seed-7 folds chose among the 8 finalists (D22) | README table, report §1: "not used for tuning or for generating candidates; used once, under a pre-registered rule, to choose among eight finalists" |
| 2 | F20: no identical fold, overlap up to 25 % | report §2, study guide part 3, EXPERIMENT_HISTORY D17: "no fold identical to a selection fold" |
| 3 | the audit rows name a test for each code bug (e.g. F10 `test_dry_run_results_must_not_be_registered`, F25 `test_predict_rejects_overflowing_prices`); Info/Note findings and the documentation/inference findings F29, F30 have none | executive summary: "each code bug has a test, while informational notes are documented without one"; study guide part 6 §6 likewise |
| 4 | Stage 1 1.7 + Stage 2 3.2 (§9) | report §4.5 LightGBM 4.9 min; column sums to 72.0 |
| 5 | `src/houseprices` 2,439 lines; pytest 217 passed | study guide part 6 intro |
| 6 | argparse in the runners | part 6 §5 names `--dry-run` and `--dry-run-rows N` per script |
| 7 | finalist CSV: E061→E081 0.11807 / 0.12147 / +0.0034 | row added to part 4 table (now eight rows) |
| 8 | E077−E076 paired: mean −0.00101, sd 0.00683; ratios 0.26 / 0.33 / 0.57 | part 4 §2 and answer 2 |
| 9 | Ridge E055 minus E056/E057 per fold 0.0029–0.0066, means 0.0048 / 0.0047 | part 7 D16 row and "Ridge as a finalist", part 2b: "in every fold, by 0.005 on average" |
| 10 | 1 − 0.11807/0.39871 = 70.4 % on selection folds (69.5 % if the seed-7 figure is used) | report §1: "on the selection folds ... (0.1181 against 0.3987)" |
| 11 | section headings: rounds 1, 2, 2b, 3, 4, 5, 6, final | README, executive summary: "seven rounds plus a final round" |

Study guide rebuilt from the drafts with `build_study_guide.py` (only the edited draft lines changed). Checks after the edits: no stale phrase remains outside this audit file; headline numbers (0.1217, 0.1207, 0.10645, 0.092–0.121, 72 of 180 min, E066 hash) unchanged; registry 83 lines; holdout log one line; originals' SHA-256 unchanged; suite 217 passed. Claim F4 of the packet closed. `docs/PENDING_FIXES.md` deleted.
