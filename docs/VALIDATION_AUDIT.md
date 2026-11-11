# Validation audit (Agent D)

## 1. Scope and method

Independent review of the binding methodology (decisions D1-D6, requirements R1-R7). Round 1 covers `src/houseprices/config.py`, `validation.py` and `registry.py`. Method: line-by-line reading, executable tests in `tests/test_validation.py` (synthetic frames with Id, GrLivArea, SalePrice, plus the real `train.csv` for the holdout), and probing sklearn 1.7.2 behaviour directly. Pipeline code is read-only for the auditor; bugs are reported, not fixed.

Test result (houseprices env, Python 3.12): `19 passed, 3 xfailed`. The three xfails are strict and pin findings F1-F3; they flip to failures when the code is fixed, which is the signal to remove the marker.

> Status: round 1 complete. Rounds 2+ (cleaning, preprocess, models, ensembles, final pipeline) are WIP / not started.

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
| R1 log1p target, RMSE on log scale, expm1 inverse | Verified | Tests for rmse_log, rmsle_dollars, round trip. Models not yet built; re-check that each pipeline is fit on `to_log_target` output. |
| R2 fitted preprocessing inside training folds | Not yet applicable | No preprocessing pipeline exists yet. Round 2: GroupMedianImputer and encoders. |
| R3 filters after split, training part only, identical validation index | Verified (runner level) | Tests d. Still to check: any future filter that uses statistics across training rows must not see validation rows (the interface guarantees this). |
| R4 holdout carved once, deterministic, seed 42, decile-stratified | Partly verified | Generator verified. JSON not yet written; no tuning-time access control exists (nothing prevents loading the holdout frame). Re-check usage log at the end. |
| R5 nested CV for stacking | Not yet applicable | No stacking code. Note F1 must be fixed before OOF from `cross_validate` is used. |
| R6 reproducibility via `evaluate --experiment` | Not yet applicable | `evaluate.py` does not exist; registry stores seed, folds, builder. |
| R7 originals untouched, no SalePrice in features, no test info in fits | Not yet applicable | Cleaning code not yet audited. Original files unchanged as far as visible; confirm against commit fedba71 in the final round. |

Cross-cutting item (added by director): dtype checks must use `select_dtypes(include=["object", "string"])` so behaviour is identical under pandas 2 and 3. Status: not yet applicable; to be checked in cleaning/preprocess rounds. Environment: pandas 2.3.3, scikit-learn 1.9.1; the validation tests were re-run there with the same result as on sklearn 1.7.2 (19 passed, 3 xfailed), so finding F1's root cause (RepeatedKFold has no `n_splits`) holds on 1.9.1 as well.

## 4. Open questions for the director

1. Should `cross_validate` reject non-finite predictions (F2) and use a correct single-pass test (F1)? Recommended yes, both are two-line changes by the code owner.
2. How will holdout access be controlled during tuning? Suggest a function that logs each call to `experiments/holdout_usage.log` and refuses a second scoring call.
3. Selection bias: with many experiments compared on the same 5 folds, small differences below the fold standard deviation are noise. Proposal: only retain changes that win on RepeatedKFold 5x3 or by more than one paired standard error.
4. `drop_large_living_area(4000)` is not equivalent to the two-Id list. On train.csv it selects four houses: Ids 524 and 1299 (low price for size) but also 692 and 1183 (prices 755,000 and 745,000, consistent with their size). Those two are legitimate extreme-value houses, so this policy removes real signal and should be registered as a distinct experiment, not as a proxy for the Id list. Rule-based variants should use the area-versus-price residual, not area alone.
