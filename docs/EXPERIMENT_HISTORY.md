# Experiment history and decision log

Chronological. Each experiment also has a machine-readable record in `experiments/registry.jsonl` (summary in `experiments/registry_summary.csv`). This file adds the reasoning, the failures and the director's rulings on disagreements.

## 2026-11-11 — Phase 0: checkpoint, environment, skeleton

- Verified ZIP backup and Git checkpoint `fedba71` before any modification (see `docs/AUDIT_2026-11-11.md`).
- Created conda env `houseprices` (Python 3.12) with core packages only; TabPFN and GPU builds excluded by user decision.
- Skeleton: `src/houseprices/{config,validation,registry}.py`, `pyproject.toml`, `docs/TASK_BOARD.md`.

### Decisions

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D1 | Target is `log1p(SalePrice)`; metric is RMSE on that scale | raw-price RMSE; `log` instead of `log1p` | Competition metric is RMSLE; `log1p`/`expm1` is the exact inverse pair. |
| D2 | Selection protocol: shuffled 5-fold, seed 42; finalists RepeatedKFold 5x3 | single split; 10-fold | 1,460 rows: 5-fold keeps ~1,170 training rows per fold; repeats quantify variance for finalists. |
| D3 | Outlier removal only filters TRAINING rows of a fold | filter whole dataset before CV | Filtering validation rows would inflate scores by deleting hard cases (user correction 2). |
| D4 | 15 % holdout, stratified by price decile, single use after freezing | none; nested CV only | Cheap unbiased check of the whole selection procedure; usage logged (user correction 3). |
| D5 | B04 implemented as a condition (all basement fields missing), not `Id == 2121` | keep Id test | Generalisable rule; the condition selects exactly 2121 in test and nothing in train (asserted by test). |
| D6 | C01 reconstructed from documentation, labelled as reconstruction | claim recovery from backup | The cell is absent from every copy; only its documented behaviour and saved verification output exist. |

## 2026-11-11 — Phase A (repair and reproduce), usage-limit checkpoint

- Agent A ported the 17 rules to `src/houseprices/cleaning.py` with the reconstructed C01 and the condition-based B04; `tests/test_cleaning.py` asserts every historical count. The repaired notebook `notebooks/01_cleaning_repaired.ipynb` executes top-to-bottom from `notebooks/` (80 cells, zero errors) and reproduces the C01 verification output (1/12 entries, 1/9 houses). Original notebook byte-identical (SHA-256 re-checked).
- Agent D round 1 found three real defects in the director's `validation.py` (F1 repeated-CV OOF overwritten, F2 NaN predictions silently scored, F3 empty training set after a filter not detected). All three were fixed by the director the same session; the strict xfail tests now pass as ordinary tests. Full suite: 77 passed.
- No experiment has been run yet. `experiments/registry.jsonl` does not exist; the first id will be E001.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D7 | pandas pinned to 2.3.3 in the research env | pandas 3.1.1 (resolved by pip) | pandas 3 changes the default string dtype; pinning keeps the repaired notebook's semantics identical to the original run. Code still uses `select_dtypes(include=["object","string"])` to be forward-compatible. |
| D8 | `cross_validate` raises on non-finite predictions and on an empty training fold | return NaN | A NaN score would be written to the registry and sorted last silently (Agent D, F2/F3). |

## 2026-11-13 — Session recovery after authentication failure

Context. A new session resumed from commit `1c67494`. The three Sonnet teammates (validation-auditor, model-researcher, feature-researcher) were spawned and each failed within minutes with `API Error: 403 Attention Required! | Cloudflare` ("Please run /login"). The user re-authenticated; the teammates were resumed one at a time (auditor first as the connectivity probe) and continued from their transcripts.

Files touched before the failure (all reviewed by the director, none discarded):

| File | Author | Change | Verdict |
|---|---|---|---|
| `src/houseprices/data.py` | director | `_with_id` re-attaches `Id` as the first column of X / X_holdout / X_test | kept: fixes a real defect (see F9 below) |
| `tests/test_data.py` | director | 4 tests: Id first and unique, Id/target alignment, outlier filter works on the feature table, feature-set parsing | kept, green |
| `experiments/models/_common.py` | director + B | `agent` kwarg on `run_and_register`; B added `set_dry_run`, `_load`, `find_registered` (config-hash lookup) | kept |
| `src/houseprices/cleaning.py` | A | `clean_train_test` now applies `add_anomaly_flags` to both frames (3 lines, additive) | kept, pending D's re-check |
| `src/houseprices/features.py` | A | new group `anomaly_flags` (never in "base") | kept; A updates the two stale shape/name tests |
| `tests/test_models.py` | B | builder smoke tests; the clone test compared parameter lists containing estimator objects / NaN | kept; B fixes the clone test |

Test state at the moment of recovery: 96 passed, 12 failed, all 12 in the two stale agent tests above; no pipeline test failed. Originals verified byte-identical to `fedba71`/`1c67494` (SHA-256: notebook 1BB32B57…, train 1E18ADDF…, test 8FDD3D82…). Line endings: director edits had converted two LF files to CRLF; normalised back before commit.

Separate issue, not fixed on purpose: a `node: command not found` hook error comes from the user-level `~/.claude/settings.json` ("pixel-agents" hooks call `node`, which is not installed on PATH). It is unrelated to the `houseprices` environment; a backup `settings.json.pixel-agents.backup` exists next to it. Left to the user.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D9 | `data.load_development_data` re-attaches `Id` as the first column after feature construction (F9, High) | make `group_base` keep Id; make the outlier filters positional | `group_base` deliberately excludes the identifier from model inputs; the training-row filters select by Id (`config.CANDIDATE_OUTLIER_IDS`) and `predict.py` needs Id for the submission. Every builder drops `Id` before preprocessing, so it never reaches an estimator. Before the fix `drop_candidate_outliers` raised `KeyError: 'Id'` on the feature table (reproduced on 2026-11-13); after the fix `tests/test_data.py` asserts exactly Ids 524 and 1299 are removed. |
| D10 | Interrupted agent work is kept and completed by its owner, not reverted | `git checkout` of the dirty files | Each partial change was reviewed and is additive; reverting would redo work. The two stale tests are the owners' to update, keeping file ownership intact. |

## Experiments

(appended as they run; format: id, hypothesis, rationale, implementation, evaluation, results, interpretation, decision)

### 2026-11-13 — E001–E004 baselines (Phase B), feature set "base", keep_all, KFold(5, shuffle, seed 42)

Run by the director with `experiments/models/run_baselines.py` after Agent D's Phase A sign-off. The first run created `experiments/holdout_ids.json` (219 ids, 15 %, stratified by price decile, seed 42); it is committed and must never be regenerated. Development set: 1,241 rows.

| id | model | CV RMSE (log1p) | fold std | fold scores | seconds |
|---|---|---|---|---|---|
| E001 | Mean baseline | 0.3987 | 0.0274 | 0.368, 0.427, 0.427, 0.378, 0.394 | 0.03 |
| E002 | OLS on numeric columns | 0.1599 | 0.0348 | 0.134, 0.217, 0.168, 0.136, 0.144 | 0.12 |
| E003 | Ridge alpha=10, one-hot | 0.1409 | 0.0316 | 0.120, 0.185, 0.163, 0.112, 0.125 | 1.15 |
| E004 | XGBoost defaults | 0.1487 | 0.0190 | 0.127, 0.158, 0.176, 0.138, 0.144 | 2.01 |

Reproducibility: `python -m houseprices.evaluate --experiment E003` and `--experiment E004` both return `reproduces_registered_scores: true` (fold scores equal to 1e-5).

Interpretation (director):
- The mean baseline (0.399) is the standard deviation of log-price in the development set; every model must be judged against it.
- Numeric-only OLS removes 60 % of that error (0.160). Adding one-hot categoricals with L2 shrinkage (Ridge alpha=10, untuned) improves to 0.141; default XGBoost on the same inputs gives 0.149 with a visibly smaller fold spread (0.019 vs 0.032).
- Fold 2 is the hard fold for every linear model (Ridge 0.185 vs 0.112–0.125 elsewhere); the two EDA outliers (Ids 524, 1299) are development rows and fall into validation there. This is exactly the case the training-fold-only outlier policy (C2) is designed to measure, and it is why selection decisions must use paired per-fold differences, not means alone.
- Decisions: all four retained as reference levels. Nothing is tuned yet; the Ridge alpha and the boosters are refined in C4/C5.

### 2026-11-13 — E005–E048: feature ablations (C1), skew correction, LotFrontage imputation (C3)

Run by Agent A (feature-researcher) with `experiments/feature_ablation/run_ablation.py` (then `--skew`) and `run_lotfrontage.py`; wall time about one minute in total (1–2 s per 5-fold CV). Models: Ridge alpha=10, ElasticNet(alpha=0.001, l1_ratio=0.5), XGBoost defaults; feature set "base" plus named groups; keep_all; KFold(5, shuffle, seed 42). E042 reproduces via `evaluate` (0.12562). Raw tables: `experiments/feature_ablation/ablation_20261113_113850.csv`, `ablation_20261113_113903.csv`.

| feature set | Ridge | ElasticNet | XGBoost |
|---|---|---|---|
| base | 0.14088 (E003) | 0.13786 (E005) | 0.14868 (E004) |
| base+areas | 0.14097 | 0.13792 | 0.14525 |
| base+baths | 0.14086 | 0.13753 | 0.14316 |
| base+ages | 0.14094 | 0.13789 | 0.14663 |
| base+quality_interactions | 0.13854 | 0.13496 | 0.13881 |
| base+ordinal_quality | 0.14120 | 0.13778 | 0.14697 |
| all = base+areas+baths+ages+quality_interactions+ordinal_quality (E021–E023) | 0.13871 | 0.13503 | 0.13562 |
| all minus areas | 0.13897 | 0.13540 | 0.13684 |
| all minus baths | 0.13871 | 0.13527 | 0.13763 |
| all minus ages | 0.13867 | 0.13505 | 0.13354 |
| all minus quality_interactions | 0.14130 | 0.13755 | 0.14422 |
| all minus ordinal_quality | 0.13834 | 0.13445 | 0.13683 |
| base, skew_log1p in-fold (E039, E040) | 0.13192 | 0.12845 | n/a |
| all, skew_log1p in-fold (E041, E042) | 0.13042 | 0.12562 | n/a |

LotFrontage imputation (base, keep_all): neighborhood median 0.14088 / 0.14868 (Ridge / XGBoost, E043, E046); global median 0.14099 / 0.14701 (E044, E047); KNN on five house-size predictors, fit in-fold, 0.14048 / 0.14900 (E045, E048).

Interpretation (Agent A, confirmed by the director):
- `quality_interactions` (OverallQual x GrLivArea, OverallQual x TotalSF) is the group that matters: removing it from "all" is the only removal that hurts every model (by 0.0025–0.009). The other groups move scores by 0.003 or less at default hyper-parameters.
- `ordinal_quality` does not help on its own; removing it from "all" improves Ridge by 0.0004 and ElasticNet by 0.0006 but makes XGBoost WORSE by 0.0012 (E038 0.13683 vs E023 0.13562). All three deltas are within fold noise. (Corrected 2026-11-13 after Agent A's review; an earlier version of this line had the XGBoost sign wrong.)
- The in-fold log1p of skewed non-negative numeric columns is the largest single gain for the linear models (about 0.009 on both base and all). Trees are invariant to monotone transforms, so it is not applied to them.
- The anomaly flags are constant (all zero) in the training data; the anomalies exist only in test rows (Ids 2593, 2550, 1877), so `base+anomaly_flags` cannot differ from base and was excluded from the run.
- LotFrontage: no imputation method is clearly better; all Ridge differences are at or below 0.0005 and the XGBoost spread (0.002) is within fold noise.
- Caveat: the XGBoost column uses default hyper-parameters; feature gains for trees may change after tuning.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D11 | Working feature set for the rest of the programme is "all" (`base+areas+baths+ages+quality_interactions+ordinal_quality`) | "all minus ordinal_quality" (best linear at default alpha), "all minus ages" (best XGBoost at defaults) | Every leave-one-out variant is within about 0.001 of "all" except the removal of quality_interactions, which hurts. Model-specific pruning at untuned hyper-parameters would be fitting to fold noise; the richest set is kept and the ordinal_quality question is re-examined on the finalists under RepeatedKFold 5x3. |
| D12 | Linear models (Ridge, Lasso, ElasticNet) always use `preprocessor_kwargs={"skew_log1p": True}`; trees never do | apply to all models; apply to none | Gains of about 0.009 for both linear models, consistent across base and all; the transformer is fit in-fold (column selection learned from training rows only, verified by Agent D in VALIDATION_AUDIT section 5.4). Trees are invariant to monotone transforms, so applying it there would only add cost. |
| D13 | LotFrontage keeps the neighbourhood-median in-fold imputation (the original notebook's semantic choice) | global median; KNN | No method differs by more than 0.002 and none exceeds the paired standard error; the default is kept because it is the documented choice of the original cleaning log and costs nothing. |
| D14 | Registry decisions "retain/reject/investigate" from the runners use a one-paired-standard-error rule on 5 folds and are provisional | decide on means | Agent D's note: differences near 0.001 are noise at 5 folds; finalists are confirmed with RepeatedKFold 5x3 before anything is frozen. |

### 2026-11-13 — E049–E054: outlier policy under training-fold-only filtering (C2)

Run by Agent B with `experiments/models/run_outlier_policy.py` on feature set "all" (D11), linear models with skew (D12), KFold(5, shuffle, seed 42). The keep_all cells reuse E041 (Ridge), E042 (ElasticNet) and E023 (XGBoost). Validation folds are always complete; the policies remove rows from the training part of each fold only (D3). Paired differences against keep_all in brackets; SE = standard deviation of the five per-fold differences / sqrt(5).

| policy | Ridge alpha=10 | ElasticNet(0.001, 0.5) | XGBoost defaults |
|---|---|---|---|
| keep_all | 0.13042 | 0.12562 | 0.13561 |
| drop_candidate_outliers (Ids 524, 1299) | 0.13480 (+0.0044, SE 0.0053) | 0.13205 (+0.0064, SE 0.0057) | 0.14062 (+0.0050, SE 0.0028) |
| drop_GrLivArea_gt_4000 (also removes 692, 1183) | 0.13603 (+0.0056, SE 0.0058) | 0.13417 (+0.0085, SE 0.0065) | 0.13641 (+0.0008, SE 0.0036) |

Interpretation (Agent B, confirmed by the director): keep_all ranks first for all three models. Dropping the two EDA outliers is worse beyond one paired SE for ElasticNet and XGBoost; the GrLivArea rule is worse beyond one SE for ElasticNet and level for XGBoost. Mechanism: Ids 1299, 692 and 1183 fall in validation fold 1 and Id 524 in fold 2, and validation folds are never filtered, so a model trained without the extreme houses is still judged on them; the linear fold standard deviation rises from about 0.023 to about 0.035. Under an honest protocol the "outlier removal" that is common in public notebooks only helps when the same rows are also removed from the validation side, which deletes the hard cases from the score.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D15 | Training-row filter is keep_all for all remaining experiments and for the final fit | drop Ids 524 and 1299; drop GrLivArea > 4000 | Table above: no policy beats keep_all for any model; three of six cells are worse beyond one paired SE, the rest within noise. The final model is therefore trained on all development rows (and later all training rows). Robustness to extreme houses is measured, not assumed. |

### 2026-11-13 — E055–E057: linear family grids (C4)

Run by Agent B with `experiments/models/run_linear_family.py` on feature set "all", keep_all, skew_log1p in-fold, both RobustScaler and StandardScaler; 136 five-fold CVs in total (Ridge 30, Lasso 26, ElasticNet 80), about one second each; only the best per family is registered, the full grid is in `experiments/models/linear_grid_base+areas+baths+ages+quality_interactions+ordinal_quality_keep_all.csv`. One ConvergenceWarning in 136 fits (ElasticNet, smallest alpha).

| id | model | best hyper-parameters | scaler | CV RMSE | fold std | fold scores |
|---|---|---|---|---|---|---|
| E055 | Ridge | alpha = 10 | standard | 0.13028 | 0.0232 | 0.113, 0.160, 0.148, 0.105, 0.126 |
| E056 | Lasso | alpha = 5.62e-4 | robust | 0.12545 | 0.0235 | 0.110, 0.155, 0.144, 0.098, 0.120 |
| E057 | ElasticNet | alpha = 1.18e-3, l1_ratio = 0.5 | robust | 0.12555 | 0.0230 | 0.109, 0.155, 0.143, 0.099, 0.122 |

Interpretation (director):
- The L1 family (Lasso, ElasticNet) beats Ridge by about 0.005, consistently in every fold: with roughly 300 one-hot plus engineered columns and 993 training rows per fold, sparsity removes rare one-hot levels that Ridge can only shrink.
- Ridge's alpha curve is flat around 10 (5.6 and 10 differ by 0.0002); Lasso has a sharp optimum near 5e-4 (alpha 1e-3 is 0.001 worse). ElasticNet's best is essentially a Lasso-like point (l1_ratio 0.5–0.9 within 0.0003).
- Scaler: robust vs standard differ by at most 0.0001 for every family. Irrelevant on this data, kept robust (default) for Lasso/ElasticNet and standard for Ridge as registered.
- Fold 2 (which holds Id 524) remains the hard fold for every linear model (0.155–0.160).
- Selection-bias caveat: 136 grid points were compared on the same five folds; the registered bests are optimistically biased by an amount that Agent D's round 4 estimates; the finalist RepeatedKFold 5x3 and the single-use holdout bound it.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D16 | Linear finalists are Lasso E056 and ElasticNet E057; Ridge E055 is kept only as an ensemble base candidate | tune Ridge further; drop Ridge | Ridge is 0.005 worse in every fold; its alpha curve is flat, so more tuning cannot close the gap. It remains a cheap, differently regularised base for the blending/stacking diversity analysis. |

### 2026-11-13 — C5 Optuna Stage 1 (PAUSED mid-run, usage limit)

Agent B started `experiments/tuning/run_optuna_stage1.py` on feature set "all", keep_all, one-hot inputs, TPESampler(seed 42), 25 trials per booster, 8 threads. The session was paused by the user while the studies were running. XGBoost completed: 25/25 trials in 2.2 min, best CV 0.12307 (better than Lasso E056 0.12545 by 0.0024, within the 0.003 Stage-2 window). The log shows best-so-far unchanged from trial 4 onward, so XGBoost does not qualify as "still improving". LightGBM, CatBoost (one-hot) and CatBoost-native state at the pause: see SESSION_CHECKPOINT §7. The Optuna SQLite study is resumable; no partial result was registered unless noted there. Stage-2 ruling deferred to the next session once `stage1_summary.json` exists.

### 2026-11-13 — Audit round 4 (tuning/grid scripts, holdout guard, selection bias) — complete, see VALIDATION_AUDIT §7

Agent D verified: all three scripts score only through `_common.run_cv` on development rows; registered Optuna `builder_kwargs` rebuild identical pipelines (hash stable); the Stage-2 rule matches correction 7; the linear grid makes exactly 136 CV calls with `skew_log1p` inside `builder_kwargs`; the 57 registry records are internally consistent; the holdout guard behaves correctly. Measured selection bias: a 15-point Ridge grid selected on the five selection folds scored 0.1319 there and about 0.1340 on five fresh partitions, so grid optimism is about +0.002, and the seed-to-seed standard deviation of a CV mean is 0.0026. Tuned-model differences below about 0.003 must not be reported as wins. Findings: **F20 (Medium)** the first repeat of RepeatedKFold(5x3, seed 42) is identical to the selection folds, so the finalist check is only two-thirds independent; **F21 (Low)** `score_holdout_once` accepts any frame and a direct `predict` on the holdout frame is not logged; F22/F23 Info (3 h cap not enforced in code; auto-"retain" is provisional). Sign-off: tuning results admissible to the registry; **finalist step C8 blocked by F20 until the repeated check uses a different seed**.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D17 | Finalist stability check uses RepeatedKFold(5x3) with seed 7 (`make_cv(5, 3, 7)`), folds disjoint from the seed-42 selection folds; implement `config.CV_SEED_FINALISTS = 7` and an `evaluate --seed` override on resume | keep seed 42 and report repeats 2–3 separately | F20: with seed 42 one of the three repeats is the selection partition, so the estimate inherits the +0.002 selection optimism. A fresh seed gives an unbiased estimate for the chosen configuration (Agent D's measurement). |
| D18 | Two test-only data-quality gaps found by Agent A are recorded as known limitations to handle before the submission file is written: the raw `GarageYrBlt` 2207 of test Id 2593 still reaches the linear models through the base features (fix candidate: fold-safe clip to the training range inside the preprocessor); test Id 2550 (5,095 sq ft Edwards partial sale) has the same profile as training outliers 524/1299, so CV cannot settle its prediction; predictions for Ids 2593, 2550 and 1877 are checked by hand | ignore | Both are invisible to CV; documenting them is required for an honest report. |

### 2026-11-13 — E058–E060: boosting Stage 1 (C5), Optuna TPE seed 42, 25 trials per booster

Run by Agent B with `experiments/tuning/run_optuna_stage1.py` on feature set "all", keep_all, one-hot inputs (shared preprocessor, no scaling), 8 threads; the run was in flight during the emergency pause and completed on its own (the SQLite study `experiments/tuning/optuna_stage1.db` holds every trial; per-trial CSVs and logs under `experiments/tuning/`). Total Stage 1 wall time 10.8 min (XGBoost 2.2, LightGBM 1.7, CatBoost 6.9). Stage-2 rule (correction 7): gap to the best model < 0.003 or best-so-far still improving over the last 10 trials.

| id | model | CV RMSE | fold std | minutes | gap to best | improving (last 10) | Stage 2 | best params |
|---|---|---|---|---|---|---|---|---|
| E058 | Optuna stage 1: xgboost best of 25 trials | 0.12307 | 0.0191 | 2.18 | 0.0036 | False | False | `{"learning_rate": 0.0392, "n_estimators": 1600, "max_depth": 3, "subsample": 0.7571, "colsample_bytree": 0.7962, "reg_alpha": 0.0015, "reg_lambda": 0.2693, "min_child_weight": 4}` |
| E059 | Optuna stage 1: lightgbm best of 25 trials | 0.12442 | 0.0191 | 1.66 | 0.0050 | True | True | `{"learning_rate": 0.0103, "n_estimators": 1600, "num_leaves": 12, "subsample": 0.7585, "subsample_freq": 1, "colsample_bytree": 0.889, "reg_alpha": 0.1435, "reg_lambda": 0.3706, "min_child_samples": 7}` |
| E060 | Optuna stage 1: catboost best of 25 trials | 0.11946 | 0.0170 | 6.93 | 0.0000 | True | True | `{"learning_rate": 0.0347, "iterations": 1700, "depth": 5, "l2_leaf_reg": 2.1282, "subsample": 0.6194, "rsm": 0.7879}` |

Interpretation (director):
- Tuned CatBoost (E060, 0.11946) is the new best single model, 0.006 better than Lasso E056 (0.12545) and 0.016 better than default XGBoost on the same features (E023 0.13562). Tuning moved XGBoost from 0.1356 to 0.1231 and LightGBM to 0.1244; all three boosters now beat every linear model.
- Shallow, slow, heavily subsampled boosters win on 993 training rows per fold: depth 3 (XGBoost), 12 leaves (LightGBM), depth 5 (CatBoost), learning rates 0.01–0.04 with 1,600–1,700 rounds, subsample about 0.75.
- Selection-bias caveat (Agent D, round 4): best-of-25 scores are optimistic by roughly 0.002–0.005; differences below 0.003 between tuned models are not wins. The finalist RepeatedKFold check uses a fresh seed (D17).

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D19 | Stage 2 for CatBoost and LightGBM only (continue the same Optuna studies to 75 trials); none for XGBoost; CatBoost-native (categoricals encoded by CatBoost) gets its own 25-trial Stage 1 first | Stage 2 for all three; skip Stage 2 | Rule of correction 7 applied to `stage1_summary.json`: XGBoost gap 0.0036 > 0.003 and flat since trial 4; LightGBM gap 0.0050 but still improving; CatBoost is the best and still improving. Budget used 10.8 min of the 3 h cap; Stage 2 capped at 60 min, native at 25 min. |

### 2026-11-13 — F20/F21 verified; finalist step unblocked

Agent D verified the director's fixes: `score_holdout_once` refuses any frame whose Id set is not the registered holdout (F21), and the finalist folds (`CV_SEED_FINALISTS = 7`) share no fold with the seed-42 selection partition (maximum validation overlap 25 % vs 20 % by chance). Independent bias check: Lasso E056 scores 0.1282 on the seed-7 folds against 0.1255 registered, i.e. +0.0027 selection optimism, consistent with the Ridge-grid estimate. Suite: 199 passed, no xfails.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D20 | The opt-in `YearRangeClipper` (preprocess `clip_years=True`) clips the UPPER bound only, learned from the training fold | clip to [min, max] | Agent A found that two-sided clipping would also move test Ids 1603/2218 (garages built 1896/1895) to the 1900 training minimum; those are real values. Only years beyond the training maximum are impossible (Id 2593, 2207 → 2010). Provably a no-op on development rows, so CV scores are unaffected. |

### 2026-11-13 — Base-model diversity (diagnostic) and premature ensembles E061–E064

Agent C ran `experiments/ensemble/run_ensembles.py --select-best` on the Stage 1 bases (E055 Ridge, E056 Lasso, E057 ElasticNet, E058 XGBoost, E059 LightGBM, E060 CatBoost; feature set "all", keep_all). The director had asked for the diagnostic only, because Agent B held the registry lock for Stage 2; the runner registered anyway (protocol breach by the ensemble-researcher, process terminated by the director once detected). The registered records follow the full nested protocol and reproduce through `evaluate`, so they are kept and labelled "Stage 1 bases"; the ensemble block is re-run after Stage 2 with the final bases (different config hashes).

Diversity diagnostic (`experiments/ensemble/diversity_20261113_120415.csv`, single-pass OOF predictions of each base, used for nothing but this table):

| pair type | residual correlation | prediction correlation |
|---|---|---|
| linear vs linear (Ridge, Lasso, ElasticNet) | 0.970–0.997 | 0.996–1.000 |
| tree vs tree (XGBoost, LightGBM, CatBoost) | 0.946–0.962 | 0.994–0.996 |
| linear vs tree | 0.823–0.862 | 0.980–0.985 |

Reading: the three linear models are near-duplicates of each other (Lasso/ElasticNet residuals correlate 0.997) and so are the three boosters; the only genuine diversity is between the two families. XGBoost and LightGBM residuals are slightly less correlated with the linear models (0.82–0.83) than CatBoost's (0.85–0.86), but CatBoost is the strongest base (OOF RMSE 0.1204 vs 0.1242 XGBoost). Expectation: a linear+tree pair captures most of the ensemble gain; adding a third model of the same family adds little.

Premature ensembles (Stage 1 bases, KFold 5 seed 42, nested where applicable):

| id | ensemble | CV RMSE | vs best base (CatBoost 0.11946) |
|---|---|---|---|
| E061 | equal-weight average Lasso + CatBoost | 0.11807 | −0.0014 |
| E062 | NNLS blend Lasso + CatBoost (weights learned in-fold) | 0.11902 | −0.0004 |
| E063 | NNLS blend of all six bases | 0.11943 | −0.0000 |
| E064 | NNLS blend of the three linear models | 0.12585 | +0.0064 (worse than CatBoost; better than Lasso 0.12545 by nothing) |

Reading: the naive 50/50 average of the best linear and the best tree model is the best result so far; learned blend weights do not beat it (the weights are fit on inner OOF predictions of about 800 rows and add variance), and blending six near-duplicate bases is no better than CatBoost alone. All differences are within the 0.003 noise band and must be confirmed on the seed-7 finalist folds.

### 2026-11-13 12:20 — Controlled pause V3 (usage limit)

State at the pause: registry E001–E064 (64 records, consistent); tests 214 passed; holdout never scored; originals unchanged. The CatBoost-native Optuna study was running under its 25-minute cap with 3 completed trials (best 0.1241) and one in progress; Stage 2 (D19) not started. Outstanding hypotheses for the next session: (H1) Stage 2 improves CatBoost by less than the 0.003 noise band; (H2) the equal-weight Lasso+CatBoost average remains the best procedure on the seed-7 finalist folds, and learned blend weights do not beat it; (H3) native-categorical CatBoost is not worth its 10x cost over one-hot CatBoost on 1,241 rows; (H4) the holdout score lands within about 0.01 of the finalist's fresh-fold CV mean. Rejected so far (with evidence): dropping outliers (E049–E054), ordinal_quality as a stand-alone gain (E010-type runs), KNN/global LotFrontage (E043–E048), scaler choice (grid CSV), Ridge as a finalist (D16), Stage 2 for XGBoost (D19), two-sided year clipping (D20).
