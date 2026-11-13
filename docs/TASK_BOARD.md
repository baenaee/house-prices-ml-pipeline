# Task board (shared task list, owned by the Research Director)

Status values: todo | in-progress | review | done | blocked. Agents request status changes by message; only the director edits this file.

| ID | Task | Owner | Depends on | Status | Evidence / notes |
|---|---|---|---|---|---|
| P0.1 | Dated ZIP of originals, hash-verified | director | – | done | `HousePricingKaglle_original_2026-11-11.zip`, 29 entries, 0 mismatches |
| P0.2 | Git repo + checkpoint commit (HTML excluded) | director | P0.1 | done | commit `fedba71`, `git status` clean |
| P0.3 | conda env `houseprices` (py3.12) + core packages, kernel registered | director | – | done | pandas pinned 2.3.3 (matches original notebook); sklearn 1.9.1, xgb 3.4.1, lgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, shap 0.54.1 |
| P0.4 | `environment.yml` / `requirements.txt` with resolved versions | director | P0.3 | done | commit `cff763b` |
| P0.5 | `docs/AUDIT_2026-11-11.md` (audit record + hash manifest) | director | P0.1 | done | |
| P0.6 | Package skeleton: config, validation, registry, data, evaluate, pyproject | director | – | done | |
| A1 | `cleaning.py`: 17 rules as pure functions + reconstructed C01 + condition-based B04 | feature-researcher | P0.6 | done | commit `564e88b`; all counts asserted |
| A2 | `tests/test_cleaning.py`: all verification counts, schema, Id/SalePrice integrity | feature-researcher | A1 | done | green |
| A3 | `notebooks/01_cleaning_repaired.ipynb`: C01 restored, duplicate removed, runs top-to-bottom from notebooks/ | feature-researcher | A1 | done | executed copy: 80 cells, 0 errors, C01 output 1/12 and 1/9; original SHA unchanged |
| A4 | Anomaly policy (flags + derived columns only), dictionary spelling map use | feature-researcher | A1 | done | flags wired into `clean_train_test`, group `anomaly_flags`, tests on Ids 2593/2550/1877; flags are all zero in train (test-only anomalies) |
| A5 | Audit of A1–A4 (no SalePrice use, no fitted statistics, no train/test concatenation) | validation-auditor | A2, A3 | done | round 2 signed off 2026-11-13 (VALIDATION_AUDIT §5): F9 High fixed, F10/F11 Low open for B; skew/KNN transformers and ensemble.py in round 2b/3 |
| B1 | `preprocess.py` + `features.py` base feature set; first pipelines | feature-researcher | A1 | done | used by E001–E004; `tests/test_models.py` exercises every builder |
| B2 | Baselines E001–E004 (mean, OLS numeric, Ridge one-hot, default XGB) | model-researcher | B1 | done | run 2026-11-13: 0.3987 / 0.1599 / 0.1409 / 0.1487; E003 and E004 reproduce via evaluate; holdout_ids.json created and committed |
| B3 | Metric/fold/holdout verification, per-fold refit check | validation-auditor | B2 | in-progress | round 1 done: F1–F3 found and fixed by director; `tests/test_validation.py` green |
| C0 | `ensemble.py` (NestedStackingRegressor, BlendingRegressor) + synthetic tests | ensemble-researcher | P0.6 | done | audit round 3 approved (VALIDATION_AUDIT §6), Low findings F16–F19 |
| C1 | Feature ablations (groups on/off, 3 models) | feature-researcher | B2 | done | E005–E042, 2026-11-13; ruling D11 (feature set "all") and D12 (skew for linear models); E042 reproduces |
| C2 | Outlier policy CV (train-fold-only filters) | model-researcher | B2 | done | E049–E054; keep_all best for all 3 models; ruling D15 |
| C3 | LotFrontage imputation comparison (in-fold) | feature-researcher | B2 | done | E043–E048; no method differs beyond noise; ruling D13 keeps neighbourhood median |
| C4 | Linear family grids (Ridge/Lasso/ElasticNet, scalers) | model-researcher | C1 | in-progress | 136-point grid on "all"+skew, keep_all |
| C5 | Boosting Stage 1: 25 Optuna trials each XGB/LGBM/CatBoost | model-researcher | C1 | in-progress | queued after C4; cap 45 min + 25 min CatBoost-native; overall tuning cap 3 h |
| C6 | Boosting Stage 2 (conditional expansion) | model-researcher | C5 | todo | director approves scope |
| C7 | Blending + nested stacking + diversity analysis | ensemble-researcher | C4, C5 | in-progress | runner being written (reads best base ids from the registry); real run after C5 |
| C8 | Finalists RepeatedKFold 5x3; freeze procedure; single holdout evaluation | director | C7 | todo | |
| E1 | Interpretability + error analysis + figures | model-researcher | C8 | todo | |
| F1 | `evaluate.py`, `train.py`, `predict.py`, end-to-end test, submission file (not submitted) | director | C8 | todo | |
| F2 | Final independent review of pipeline and claims | validation-auditor | F1 | todo | |
| D1 | README, RESEARCH_REPORT, STUDY_GUIDE, EXPERIMENT_HISTORY, CHATGPT_AUDIT_PACKET, EXECUTIVE_SUMMARY | director (+B, C review) | F2 | todo | mandatory deliverables |

| R1 | Recovery audit after 2026-11-13 Cloudflare 403 failure: originals hashed, dirty files reviewed, tests run | director | – | done | see EXPERIMENT_HISTORY 2026-11-13 entry; 96 passed / 12 stale agent tests failing at recovery |
