# Session checkpoint — 2026-11-11 (usage-limit stop)

Purpose: let the next Claude session continue the research without restarting it. Read this file, then `docs/TASK_BOARD.md`, `docs/EXPERIMENT_HISTORY.md`, `docs/VALIDATION_AUDIT.md`, and the approved plan at `C:\Users\Enrique Baena\.claude\plans\pasted-content-id-b077-create-an-smooth-charm.md`.

## State of the repository

- Git (installed this session at `C:\Program Files\Git\cmd\git.exe`; not on PATH in Claude's shell, call it by full path). Commits: `fedba71` original checkpoint → `00ae167` skeleton → `cff763b` environment → Phase A WIP commit (see `git log`).
- Original files are byte-identical to the checkpoint (verified by SHA-256 after all agent work): `HousePricing.ipynb` 1BB32B57…, `house_prices_training_profile.html` 2A97F907… (git-ignored, preserved in the ZIP), CSVs unchanged.
- Full backup: `C:\Users\Enrique Baena\Desktop\HousePricingKaglle_original_2026-11-11.zip` (29 entries, hash-verified).
- Environment: conda env `houseprices`, python `C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe` (3.12; pandas pinned 2.3.3, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, shap 0.54.1). Package installed editable. Jupyter kernel `houseprices` registered. `requirements.txt` = pip freeze; `environment.yml` wraps it. Plain `python` is NOT on PATH (Microsoft Store alias).
- Test suite: `& "<env python>" -m pytest tests -q` → all passing (62 passed, 3 xfail) at checkpoint time.

## What is done

Phase 0 complete. Phase A code complete and green, pending Agent D's audit sign-off:
- `src/houseprices/cleaning.py` — the 17 rules as pure functions, identical for train/test, with the RECONSTRUCTED C01 (labelled as such) and B04 as a condition (all basement fields missing), not `Id == 2121`.
- `tests/test_cleaning.py` — every notebook verification count asserted (see `docs/AUDIT_2026-11-11.md` table), schema/Id/SalePrice integrity, idempotency.
- `src/houseprices/features.py`, `src/houseprices/preprocess.py`, `tests/test_features.py` — feature groups and fold-safe preprocessor (GroupMedianImputer for LotFrontage).
- `notebooks/01_cleaning_repaired.ipynb` + `01_cleaning_repaired.executed.ipynb` — runs top-to-bottom from `notebooks/` (80 cells, execution counts 1–70, zero errors); C01 verification output 1/12 entries, 1/9 houses; final check 0 categorical NaN, Ids and SalePrice preserved. Original notebook untouched.
- Director modules: `config.py`, `validation.py` (log1p target, rmse_log, seeded folds, single-use holdout, training-fold-only row filters), `registry.py` (JSONL experiment registry), `data.py` (single data entry point), `evaluate.py` (`python -m houseprices.evaluate --experiment <id>`).
- `docs/VALIDATION_AUDIT.md` round 1 by Agent D; `tests/test_validation.py`.

## In progress at the moment of the stop (see agent reports section below once filled)

- Agent B (`model-researcher`): writing `src/houseprices/models.py` and `experiments/models/run_baselines.py`. NO experiment has been run; the registry `experiments/registry.jsonl` does not exist yet (E001 will be the first id).
- Agent C (`ensemble-researcher`): writing `src/houseprices/ensemble.py` (NestedStackingRegressor, BlendingRegressor) and `tests/test_ensemble.py` on synthetic data. No real-data runs.
- Agent D (`validation-auditor`): round 1 on validation/config/registry done; round 2 (audit of cleaning/features/preprocess) not yet requested.
- Agent A (`feature-researcher`): A1–A3 done; A4 (anomaly flags/ages) and B1 refinements may be partially done — check `git diff` and the agent report below.

## Exact next steps (in order)

1. Re-create teammates if the session is new (names: feature-researcher, model-researcher, ensemble-researcher, validation-auditor; Sonnet; prompts summarised in the plan file and TASK_BOARD). Give each `docs/SESSION_CHECKPOINT.md` to read first.
2. Agent D round 2: audit `cleaning.py` (no SalePrice use, no fitted statistics, train/test identical, B04 condition, C01 labelled reconstruction), `features.py`, `preprocess.py` (GroupMedianImputer fit-only-on-fold). Sign off `docs/VALIDATION_AUDIT.md` §Phase A.
3. Confirm `models.py` imports and builders smoke-test on synthetic data; then run `experiments/models/run_baselines.py` → E001–E004 (mean, OLS numeric, Ridge one-hot, default XGBoost) on feature_set "base". Confirm `python -m houseprices.evaluate --experiment E003` reproduces.
4. Feature ablations (Agent A), outlier-policy CV (Agent B, training-fold-only filters), LotFrontage imputation comparison, linear-family grids.
5. Boosting Stage 1: 25 Optuna trials each (XGB, LGBM, CatBoost), cap 45 min; Stage 2 only if justified, overall tuning cap 3 h.
6. Ensembles via `ensemble.py` nested stacking / blending (Agent C), diversity analysis.
7. Finalists RepeatedKFold 5x3; freeze; single holdout evaluation; `train.py`/`predict.py`; end-to-end test; submission file (NOT submitted).
8. Documentation deliverables (README, RESEARCH_REPORT, STUDY_GUIDE, EXPERIMENT_HISTORY, CHATGPT_AUDIT_PACKET, EXECUTIVE_SUMMARY).

## How to avoid duplicating work

- The registry is append-only; before running an experiment, check `experiments/registry_summary.csv` for an identical `config_hash`.
- Holdout ids are fixed in `experiments/holdout_ids.json` once created; never regenerate with a different seed.
- Do not regenerate `house_prices_training_profile.html`.

## Standing limits

No Kaggle submissions; no TabPFN; no destructive change to original files; no paid services; tuning cap 3 h CPU; approval needed before anything outside the approved plan.

## Agent wrap-up reports (received 2026-11-11, all work saved and committed)

**Agent A, feature-researcher — complete, nothing in progress.**
- `cleaning.py`: 17 rules, `clean`, `clean_train_test`, `load_raw`, `basement_fully_unrecorded`, `add_anomaly_flags`. `features.py`: `FEATURE_SETS` with six groups (base, areas, baths, ages, quality_interactions, ordinal_quality) and `build_features`. `preprocess.py`: `GroupMedianImputer`, `split_columns` (any non-numeric column is categorical, pandas 2/3 safe), `make_preprocessor(X_sample, ...)`.
- Tests: 30 cleaning + 13 features = 43 passed. All 17 counts match; deferred NaN match; B04 selects only Id 2121; cleaning idempotent. Repaired notebook executed: 80 cells, 0 errors.
- Deviations from spec (accepted by the director): `make_preprocessor` takes `X_sample` (a DataFrame) rather than a column list; `ordinal_quality` adds `<col>_ord` columns and keeps the text columns; anomaly flags are excluded from "base".
- Director decisions on A's questions: anomaly flags MAY be tested as an extra feature group ("anomaly_flags") in the ablation, never in "base"; an ablation runner IS wanted at `experiments/feature_ablation/run_ablation.py` (Agent A writes it against `validation.cross_validate`, using Agent B's `build_ridge`, `build_elasticnet`, `build_xgboost` builders).

**Agent B, model-researcher — code complete, NOTHING RUN.**
- `models.py`: builders mean, OLS numeric, ridge, lasso, elasticnet, xgboost, lightgbm, catboost (`native_categoricals` kwarg); `OPTUNA_SPACES`; `BUILDERS`; helpers `LazyPreprocessor` (builds the preprocessor from the training fold's columns inside fit; reads no values), `NumericMedianImputer`, `CatBoostNative`. Linear builders accept `scaler="robust"|"standard"`. Tree builders default to `scale=False`. No early stopping; `n_estimators` is tuned.
- `experiments/models/_common.py` (`run_and_register`) and `run_baselines.py` (E001–E004; refuses to run if ids exist). Smoke-tested on synthetic data only.
- Tuner note: call `OPTUNA_SPACES[name](trial)` to get params; do not rebuild from `trial.params` (LightGBM space fixes `subsample_freq=1` outside the trial).
- Resume: after D's round-2 sign-off, `& "<env python>" experiments\models\run_baselines.py`, then `python -m houseprices.evaluate --experiment E003`.

**Agent C, ensemble-researcher — C0/C1 complete, C2 not started.**
- `ensemble.py`: `NestedStackingRegressor`, `BlendingRegressor` (weights learned inside fit when `weights=None`), `fit_blend_weights` (NNLS / SLSQP), `inner_oof_matrix`, `prediction_correlation`, `residual_correlation`, `diversity_report`, `build_blend`, `build_stack`, plus labelled optimistic diagnostics `optimistic_oof_stack_score`, `in_sample_stack_score`. Nesting is inside `fit()`, so `validation.cross_validate` remains the single outer protocol. 12 tests passed, including the memoriser leak test.
- Director decisions on C's questions: Stage 1 uses `passthrough=False` only (no numeric-matrix plumbing); Ridge meta-learner only; nested cost accepted (minutes per stack).
- Resume: needs the registry ids of the best Ridge/Lasso/ElasticNet/XGBoost/LightGBM/CatBoost from Agent B before writing `experiments/ensemble/run_ensembles.py`.

**Agent D, validation-auditor — round 1 complete.**
- F1–F3 (medium/low) in `validation.py`: FIXED by the director; tests converted from strict xfail to ordinary tests (77 passed).
- F4 (low): holdout file trusted without validation → FIXED (ids checked against the data, duplicates rejected).
- F5 (low): `config.DICTIONARY_TO_DATA_VALUES["Exterior2nd"]` key misspelled `CmentBd` → FIXED to `CemntBd` (dictionary) → `CmentBd` (data).
- Q4 (noted): `drop_large_living_area(4000)` also removes Ids 692 and 1183 (legitimate expensive houses); it is a deliberately different policy from `drop_candidate_outliers`, both are evaluated under CV.
- Round 2 (cleaning/features/preprocess audit) NOT yet requested; must precede the baseline runs.

## Decisions pending the user (none blocking)

None. All approvals needed for the next steps were granted in the plan. The next session should begin with Agent D's round 2, then E001–E004.
