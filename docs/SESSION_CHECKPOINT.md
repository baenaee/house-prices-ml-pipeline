# Session checkpoint — 2026-11-13 (emergency pause, usage limit)

Purpose: let the next Claude session continue the research without restarting it. Read this file, then `docs/TASK_BOARD.md`, `docs/EXPERIMENT_HISTORY.md` (decisions D1–D16 are binding), `docs/VALIDATION_AUDIT.md`, the approved plan at `C:\Users\Enrique Baena\.claude\plans\pasted-content-id-b077-create-an-smooth-charm.md`, and paste `docs/RESUME_PROMPT.md` to start. The previous checkpoint (2026-11-11) is superseded; its content is folded in below.

## 1. Repository state

- Git: `C:\Program Files\Git\cmd\git.exe` (not on PATH). Commit sequence this session: `651fb01` recovery + audit round 2 → `582bf86` baselines + holdout → `4b9ff5d` audit addendum → `0898b38` ablations → `235d730` holdout guard → `da61efd` outlier policy → `de19412` notebook 02 → `87de393` ensemble runner → `d337e09` study-guide draft 3 → `37cc669` linear grids → FINAL checkpoint commit (see `git log -1`). Working tree clean after the final commit.
- Originals byte-identical to `fedba71` (SHA-256 verified 2026-11-13: `HousePricing.ipynb` 1BB32B57…, `train.csv` 1E18ADDF…, `test.csv` 8FDD3D82…). Backup ZIP: `C:\Users\Enrique Baena\Desktop\HousePricingKaglle_original_2026-11-11.zip`.
- Environment: conda env `houseprices`, python `C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe` (3.12; pandas 2.3.3 pinned, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, shap 0.54.1). Plain `python` is NOT on PATH. `requirements.txt`/`environment.yml` unchanged.
- Tests: `& "<env python>" -m pytest tests` (pyproject sets `-q`; passing `-q` again hides the summary line) → **191 passed, 1 xfailed** at pause time (the xfail is Agent D's strict pin of an open finding in round 4, see §5).
- Known tooling quirk: writing files through Bash heredocs collapses doubled backslashes; build escapes with `chr()` or use the Edit/Write tools.
- Separate, NOT fixed on purpose: `node: command not found` hook error comes from user-level `~/.claude/settings.json` (pixel-agents hooks); unrelated to this project.

## 2. Completed phases and results

Phase 0, Phase A (audited, signed off §5 of VALIDATION_AUDIT), Phase B baselines, C1 ablations, C2 outlier policy, C3 LotFrontage, C4 linear grids are DONE. Registry: `experiments/registry.jsonl` has **57 experiments (E001–E057)** plus whatever Stage 1 registered after E057 (see §4 and §7). Holdout: `experiments/holdout_ids.json` (219 ids, seed 42, decile-stratified) created 2026-11-13 11:36 and committed; `experiments/holdout_usage.log` does NOT exist (holdout never scored).

Best validated scores so far (KFold 5, shuffle, seed 42, development set 1,241 rows, RMSE on log1p):

| id | model | feature set | CV mean | fold std |
|---|---|---|---|---|
| E001 | mean baseline | base | 0.3987 | 0.027 |
| E003 | Ridge a=10 | base | 0.1409 | 0.032 |
| E004 | XGBoost defaults | base | 0.1487 | 0.019 |
| E023 | XGBoost defaults | all | 0.1356 | – |
| E041 | Ridge a=10 + skew | all | 0.1304 | – |
| E042 | ElasticNet(0.001,0.5) + skew | all | 0.1256 | – |
| E055 | Ridge a=10 (grid best) + skew | all | 0.13028 | 0.0232 |
| E056 | **Lasso a=5.62e-4 + skew (best single model registered)** | all | **0.12545** | 0.0235 |
| E057 | ElasticNet a=1.18e-3, l1=0.5 + skew | all | 0.12555 | 0.0230 |
| Stage 1 XGBoost | tuned, 25 trials (log: best 0.12307, 2.2 min) | all | 0.12307 | see §7 for registration status |

"all" = `base+areas+baths+ages+quality_interactions+ordinal_quality` (pass the full string as feature_set; `data.py` does not know the alias). E003, E004, E042 reproduce exactly through `python -m houseprices.evaluate --experiment <id>`.

Binding rulings this session (details in EXPERIMENT_HISTORY): D9 Id re-attached in `data.py`; D11 feature set "all"; D12 linear models use `preprocessor_kwargs={"skew_log1p": True}`, trees do not; D13 LotFrontage neighbourhood median kept; D14 one-SE registry decisions are provisional; D15 `train_row_filter=keep_all` everywhere (dropping outliers hurts under complete validation folds); D16 linear finalists Lasso E056 / ElasticNet E057, Ridge kept as ensemble base.

## 3. Code written this session (all committed, all tested)

- `src/houseprices/data.py`: `_with_id` (Id first column of X/X_holdout/X_test). `validation.py`: `score_holdout_once`, `holdout_usage`, `HoldoutAlreadyUsedError`, `HOLDOUT_USAGE_LOG`. `cleaning.py`: `clean_train_test` applies `add_anomaly_flags`. `features.py`: group `anomaly_flags` (all zero in train; test-only anomalies Ids 2593, 2550, 1877). `preprocess.py`: `SkewLog1pTransformer`, `KNNLotFrontageImputer`, kwargs `skew_log1p`, `lotfrontage="knn"`. `models.py`: F11 fix (non-numeric = text). `ensemble.py`: `build_blend_from_specs`, `build_stack_from_specs`, `build_average_from_specs`, `OptimisticScore`.
- `experiments/models/_common.py`: `run_and_register(..., agent=)`, `set_dry_run` (registration refused in dry-run), `find_registered`. Scripts: `run_baselines.py` (run), `run_outlier_policy.py` (run), `run_linear_family.py` (run), `experiments/tuning/run_optuna_stage1.py` (partially run, §4), `experiments/feature_ablation/run_ablation.py` + `run_lotfrontage.py` + `_ablation_common.py` (run), `experiments/ensemble/run_ensembles.py` (NOT run; dry-run verified).
- Tests: `tests/test_data.py`, `test_holdout_guard.py` (director), `test_models.py` (B), `test_audit_round2.py` (D), extended `test_features.py`, `test_ensemble.py`.
- Notebooks: `notebooks/02_feature_engineering.ipynb` + executed copy (0 errors). Drafts: `docs/drafts/STUDY_GUIDE_part3_ensembles.md` (2,266 words; dry-run correlation figures to replace after the real ensemble run), `docs/drafts/STUDY_GUIDE_part1_data_features.md` (incomplete, see §7).

## 4. Interrupted at the pause

- **C5 Optuna Stage 1** (Agent B): command running at pause: `run_optuna_stage1.py --feature-set base+areas+baths+ages+quality_interactions+ordinal_quality --train-row-filter keep_all` (boosters xgboost, lightgbm, catboost one-hot; 25 trials each; cap 45 min), to be followed by a fourth study `--boosters catboost --catboost-native` (25 trials, cap 25 min). XGBoost finished (25/25, best 0.12307, 2.2 min). State: `experiments/tuning/optuna_stage1.db` (SQLite, gitignored, resumable, skips completed trials), `stage1_<booster>_<fs>.log`, `stage1_<booster>_<fs>_trials.csv`, `stage1_summary.json` (written at the end of a full run). Resume by re-running the same command; the script reuses the study and registers each booster's best once its study is complete. Exact state at the pause: §7.
- **Audit round 4** (Agent D): tuning/grid scripts + holdout guard; IN PROGRESS, not signed off (VALIDATION_AUDIT §7).
- **Study guide part 1 draft** (Agent A): incomplete (§7).
- Not started: C6 Stage 2 (director decides from `stage1_summary.json`), C7 ensembles (runner ready), C8 finalists RepeatedKFold 5x3 + freeze + single holdout use, E1 interpretability, F1 `train.py`/`predict.py`/end-to-end test/submission file (not submitted), F2 final audit, D1 documentation.

## 5. Open findings (VALIDATION_AUDIT)

F11 fixed/verified; F12–F15 Info (filter name int vs float; knn silent fallback; registry lacks git commit/library versions; CONSTANT_IN_TRAIN_FLAGS chosen from train); F16 Low memoriser-test blind spot (probe added by D); F17 Low manual registry path (`registry.append` has no CVResult check; `OptimisticScore` mitigates); F18 Info blend weights on same folds; F19 Info passthrough unreachable. Round 4 findings: §7 of the audit file and D's reply in §7 below. Selection-bias caveat: all grid/Optuna selection is on the same 5 folds; finalists need RepeatedKFold 5x3 and the single holdout.

## 6. Exact next steps (in order)

1. Recreate the team (see RESUME_PROMPT). Verify §1 facts.
2. Agent B: resume Stage 1 (same command) until all four studies complete; report `stage1_summary.json`; register best per booster. Director rules on Stage 2 (within 0.003 of the best model or still improving; overall tuning cap 3 h, about 2 h 50 min remain).
3. Agent D: finish round 4 and sign off tuning admissibility; then audit `run_ensembles.py` output when it runs.
4. Agent C: `run_ensembles.py --select-best --feature-set base+areas+baths+ages+quality_interactions+ordinal_quality --train-row-filter keep_all --budget-minutes 40` (needs registered lasso/elasticnet/ridge/xgboost/lightgbm/catboost on that feature set; give the registry lock to C).
5. Director: finalists (top 5 incl. best single and best ensemble) under `evaluate --repeats 3` (RepeatedKFold 5x3); freeze; record; ONE call to `validation.score_holdout_once` with the frozen id; `train.py`/`predict.py`; `tests/test_pipeline_end_to_end.py`; submission file in `outputs/submissions/` (not submitted).
6. Agent B: interpretability (permutation importance, SHAP on the tree finalist, residuals by price decile / GrLivArea) → `outputs/figures/`, `notebooks/03_modeling.ipynb`, `04_ensembles.ipynb`, `05_interpretability.ipynb` (thin).
7. Documentation: merge `docs/drafts/*` into `docs/STUDY_GUIDE.md`; write README.md, `docs/RESEARCH_REPORT.md`, `docs/CHATGPT_AUDIT_PACKET.md` (D co-writes claims), `docs/EXECUTIVE_SUMMARY.md`; final audit round (F2); final commit.

## 7. Agent wrap-up reports at the pause

(appended below as received)

**Director's observation of Stage 1 state at the cut-off (from the logs; agents' own replies did not arrive before the usage limit):**
- xgboost: 25/25 trials, best 0.12307, 2.2 min. lightgbm: 25/25 trials, best 0.12442, 1.7 min. catboost (one-hot): RUNNING at the cut-off, 5+ trials done, best-so-far 0.12052 at 11:53 (process may have continued or been interrupted by Agent B; either way the SQLite study holds every completed trial). catboost-native: not started.
- NOTHING from Stage 1 is registered yet (registry still 57 lines); the script registers each booster's best only when the whole run finishes. Resume with the same command; completed trials are skipped. Then `stage1_summary.json` gives the Stage-2 rule evaluation. All three boosters are within 0.003 of the best model so far (CatBoost 0.1205 provisional best overall), so expect Stage 2 for at least CatBoost.
- Agent D's round 4 (`docs/VALIDATION_AUDIT.md` §7, `tests/test_audit_round2.py`) and Agent A's draft `docs/drafts/STUDY_GUIDE_part1_data_features.md` were saved in whatever state they reached and are committed; both are marked/must be treated as IN PROGRESS. Notebook 02 was re-executed by Agent A (0 errors at last check); re-verify on resume.
