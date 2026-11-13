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
