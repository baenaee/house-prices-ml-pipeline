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
- `ordinal_quality` does not help on its own and its removal from "all" ties or slightly improves every model (0.0004–0.0012), within one paired standard error for Ridge and XGBoost.
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
