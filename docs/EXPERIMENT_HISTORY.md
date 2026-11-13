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
