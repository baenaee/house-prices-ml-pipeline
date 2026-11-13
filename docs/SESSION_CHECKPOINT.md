# Session checkpoint — 2026-11-13, 12:20 (controlled pause V3, usage limit)

Purpose: let the next Claude session continue without restarting. Read this file, then `docs/TASK_BOARD.md`, `docs/EXPERIMENT_HISTORY.md` (decisions D1–D20 binding), `docs/VALIDATION_AUDIT.md` (§5–§8, F1–F28), `README.md`, the approved plan at `C:\Users\Enrique Baena\.claude\plans\pasted-content-id-b077-create-an-smooth-charm.md`, and start the new session with `docs/RESUME_PROMPT.md`. Earlier checkpoints are superseded.

## 1. Repository state

- Git: `C:\Program Files\Git\cmd\git.exe` (not on PATH). Commits this session (2026-11-13), oldest first: `651fb01` recovery + audit round 2 · `582bf86` baselines + holdout · `4b9ff5d` · `0898b38` ablations · `235d730` holdout guard · `da61efd` outlier policy · `de19412` notebook 02 · `87de393` ensemble runner · `d337e09` · `37cc669` linear grids · `899f845`/`7156f9e`/`a01d426` emergency checkpoint 1 · `62b3b87` Stage 1 + D17/F21 · `8e1e257` train/predict · `b6c0f46` F20/F21 verified · `d2e4ac7` finalists script · `2f1b35d` year clipper · `7a87dd1` ensembles E061–E064 · `e1fba94` · `5e40924` F24–F26 fixes · `d2756bf` README draft · `e7fcbb8` · `f851604` F24–F26 verified · FINAL pause commit (see `git log -1`). Working tree clean after it.
- Originals byte-identical to `fedba71` (SHA-256 re-verified 12:20: notebook 1BB32B57…, train 1E18ADDF…, test 8FDD3D82…). Backup ZIP on the Desktop (2026-11-11).
- Environment: conda env `houseprices`; python `C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe` (3.12; pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, shap 0.54.1). Plain `python` not on PATH.
- Tests: `& "<env python>" -m pytest tests` → **214 passed, 0 failed, 0 xfail** at the pause (12:21). Do not add `-q` (pyproject already sets it; the summary line disappears).
- Tooling notes: Bash heredocs in this harness collapse doubled backslashes and sometimes fail on long content; write scripts to the scratchpad with the Write tool and run them. User-level `~/.claude/settings.json` has pixel-agents hooks calling `node` (not installed) → harmless hook error, not ours.

## 2. Completed phases and results

Done and audited: Phase 0; Phase A (cleaning/features/preprocess, §5 sign-off); Phase B baselines E001–E004; C1 ablations E005–E042; C3 LotFrontage E043–E048; C2 outlier policy E049–E054; C4 linear grids E055–E057; C5 Stage 1 one-hot boosters E058–E060; premature ensembles on Stage 1 bases E061–E064 (valid, see §4); diversity diagnostic; audit rounds 1–5 (F1–F28 all closed or accepted: F27 non-atomic registry, F28 pickles); F1 tooling `train.py`/`predict.py` + end-to-end test; C8 tooling `experiments/finalists/run_finalists.py` (not run).

Registry: `experiments/registry.jsonl`, **64 experiments, latest id E064** (unless the capped native study registered E065 after 12:20, see §4), no duplicate ids or hashes, each `cv_mean` equals the mean of its fold scores (auditor check). Holdout: `experiments/holdout_ids.json` (219 ids, seed 42, decile-stratified), **never scored**: `experiments/holdout_usage.log` does not exist.

Best validated so far (KFold 5, shuffle, seed 42, 1,241 development rows, RMSE log1p; all PRELIMINARY: selection-fold scores, optimistic by about +0.002–0.003 per audit §7; nothing is frozen):

| rank | id | model | CV mean | note |
|---|---|---|---|---|
| 1 | E061 | equal-weight average Lasso E056 + CatBoost E060 | 0.11807 | Stage 1 bases; within 0.003 of CatBoost alone |
| 2 | E062 | NNLS blend Lasso + CatBoost (weights learned in-fold) | 0.11902 | |
| 3 | E063 | NNLS blend of six bases | 0.11943 | no better than CatBoost alone |
| 4 | **E060** | **CatBoost, Optuna Stage 1 best (best single model)** | **0.11946** | depth 5, lr 0.035, 1700 it |
| 5 | E058 | XGBoost Stage 1 | 0.12307 | no Stage 2 (gap 0.0036, flat) |
| 6 | E059 | LightGBM Stage 1 | 0.12442 | Stage 2 approved |
| 7 | E056 | Lasso alpha 5.6e-4 + skew (best linear) | 0.12545 | fresh-fold (seed 7) check: 0.1282 |
| 8 | E057 / E055 | ElasticNet 0.12555 / Ridge 0.13028 | | |
| – | E001 | mean baseline | 0.3987 | |

"all" feature set = `base+areas+baths+ages+quality_interactions+ordinal_quality` (pass the full string; `data.py` has no alias). Reproduced via `evaluate`: E003, E004, E042, E056 (seed 7 variant), finalist probe. Diversity (`experiments/ensemble/diversity_20261113_121446.csv`): residual correlation linear–linear 0.970–0.997, tree–tree 0.946–0.962, linear–tree 0.823–0.862.

Binding rulings this session: D9 Id re-attached; D11 feature set "all"; D12 skew for linear models only; D13 LotFrontage neighbourhood median; D14 one-SE decisions provisional; D15 keep_all; D16 linear finalists Lasso/ElasticNet; D17 finalist seed 7; D18 test-only data gaps (Ids 2593, 2550, 1877); D19 Stage 2 for CatBoost + LightGBM only, native CatBoost own Stage 1; D20 year clipper upper bound only (opt-in `clip_years`).

## 3. Code and documents written this session (all committed)

`src/houseprices/`: `data.py` (`_with_id`), `validation.py` (`score_holdout_once` + `_assert_is_holdout_frame` + `holdout_usage`), `config.py` (`CV_SEED_FINALISTS = 7`), `evaluate.py` (`--seed`), `train.py` (NEW; refuses `--rows all` until holdout scored), `predict.py` (NEW; validated submission, never submits), `cleaning.py` (anomaly flags wired), `features.py` (`anomaly_flags` group), `preprocess.py` (`SkewLog1pTransformer`, `KNNLotFrontageImputer`, `YearRangeClipper`, kwargs `skew_log1p`, `lotfrontage="knn"`, `clip_years`), `models.py` (F11), `ensemble.py` (`*_from_specs` builders, `OptimisticScore`). `experiments/models/_common.py` (`agent=`, `set_dry_run`, `find_registered(seed=)`), runners: baselines, outlier policy, linear family, `tuning/run_optuna_stage1.py`, `feature_ablation/*`, `ensemble/run_ensembles.py` (`--diagnostics-only` fixed), `finalists/run_finalists.py`. Tests: `test_data.py`, `test_holdout_guard.py`, `test_pipeline_end_to_end.py`, `test_models.py`, `test_audit_round2.py`, extended features/ensemble tests. Notebook `02_feature_engineering` (+executed, 0 errors). `README.md` draft (status line says in progress). Drafts in `docs/drafts/`: `STUDY_GUIDE_part1_data_features.md` (5,730 w, complete), `STUDY_GUIDE_part2a_validation.md` (4,159 w, complete), `STUDY_GUIDE_part2b_linear_models.md` (Agent A, state in §7), `STUDY_GUIDE_part3_ensembles.md` (2,266 w; Agent C updating with real diversity numbers, state in §7), `CHATGPT_AUDIT_PACKET_claims.md` (Agent D, state in §7).

## 4. Running / interrupted / pending at the pause

- **Optuna** (`experiments/tuning/optuna_stage1.db`, SQLite, gitignored, resumable): xgboost 25/25 COMPLETE (0.12307, registered E058); lightgbm 25/25 (0.12442, E059); catboost one-hot 25/25 (0.11946, E060); **catboost_native: 3 COMPLETE + 1 RUNNING at 12:20, best 0.12411, time-capped at 25 min (started 11:59:19)**. Each native trial takes several minutes. The script registers the study's best when it stops at the cap → a possible E065 "catboost_native best of N trials" with N small; treat it as a time-capped partial result, not a tuned model (Agent B's final word in §7). Stage 2 (CatBoost, LightGBM to 75 trials, D19) NOT started. Resume command (same study names, skips completed trials): `& "<env python>" experiments\tuning\run_optuna_stage1.py --feature-set base+areas+baths+ages+quality_interactions+ordinal_quality --train-row-filter keep_all --boosters catboost --catboost-native --n-trials 25 --time-cap-minutes 25`; Stage 2: same script with `--boosters catboost,lightgbm --n-trials 75 --time-cap-minutes 60` (Agent B to confirm the exact flags; the script's `--help` is authoritative). Tuning time used: 10.8 min + native (≤25 min) of the 3 h cap.
- **Ensembles**: runner ready; the registering run (`run_ensembles.py --select-best --feature-set <all> --train-row-filter keep_all --budget-minutes 40`) must wait for Stage 2 bases and the director's GO; E061–E064 (Stage 1 bases) are already registered and will be skipped by hash.
- **Finalists (C8)**: not run. Candidate ids at the pause: E061, E062, E060, E056, E058 (+ Stage 2 bests and new ensembles). Command: `experiments\finalists\run_finalists.py --ids <csv ids>` (seed 7 default).
- **Holdout**: not scored. Procedure not frozen. `train --rows all`, submission file, interpretability (E1), notebooks 03–05, final docs: pending.

## 5. Open validation concerns

None blocking. Accepted limitations: F27 (registry append not atomic; single writer by process), F28 (joblib pickles), 219-row holdout cannot rank models closer than about 0.01, anomaly flags constant in train, test Id 2550 irreducible. Standing rule: differences < 0.003 CV RMSE are not wins; finalists on seed 7.

## 6. Exact next steps (in order)

1. Recreate the team (RESUME_PROMPT §3); verify §1 facts; check `tasklist` for stale processes; check whether E065 (catboost_native, time-capped) exists and document it as partial.
2. Agent B (registry lock): resume/finish the catboost_native study only if cheap (each trial several minutes; the director may rule to stop at the capped result), then Stage 2 for catboost and lightgbm to 75 trials (cap 60 min), registering a Stage-2 best per booster only if it beats Stage 1. Report `stage1_summary.json`-style stats.
3. Agent C (registry lock): `run_ensembles.py --select-best ...` with the final bases; report table with paired SE vs best base; D audits the output.
4. Director: `run_finalists.py --ids <top 5 incl. best single, best ensemble, best linear>`; rule the finalist; FREEZE (record in EXPERIMENT_HISTORY with id and hash); `train --experiment <id>` (development rows; add `--override '{"preprocessor_kwargs": {"clip_years": true}}'` only for pipelines whose builder accepts preprocessor_kwargs, for ensembles the clip must be inside each base spec); ONE call to `validation.score_holdout_once` with the frozen id; record; then `train --rows all`, `predict`, submission to `outputs/submissions/` (not submitted), check Ids 2593/2550/1877 by hand.
5. Agent B: interpretability (permutation importance, SHAP on CatBoost finalist, residuals by decile/GrLivArea), figures to `outputs/figures/`, thin notebooks 03–05; study-guide boosting chapter draft (part 2c).
6. Director: merge drafts into `docs/STUDY_GUIDE.md`; finish README, `RESEARCH_REPORT.md`, `CHATGPT_AUDIT_PACKET.md` (from D's claims draft), `EXECUTIVE_SUMMARY.md`; Agent D final audit round (F2); final commit.

## 7. Teammate reports at the pause

(appended below as received; if a teammate's report is missing, its last known state is in §3/§4)

- **ensemble-researcher (12:20): stopped, nothing running, no registry writes after E064.** `docs/drafts/STUDY_GUIDE_part3_ensembles.md` is COMPLETE (2,992 words): real diversity tables from `diversity_20261113_121446.csv`, per-base RMSE, new section 8b comparing E061–E064 against CatBoost E060 with paired differences (E061 −0.0014 ± 0.0019; E062 −0.0004 ± 0.0016; E063 −0.00003 ± 0.0019; E064 rejected vs Lasso). Takeaway: no ensemble clears the one-SE bar yet; the naive average edges the learned blend within noise. Standing by for the GO (after Stage 2) to run `run_ensembles.py --select-best` with final bases; it will verify edits by grep and dry-run first.
- **feature-researcher (12:20): stopped, nothing running, no registry writes.** `docs/drafts/STUDY_GUIDE_part2b_linear_models.md` COMPLETE (3,939 words; Agent B to review: alpha-scale remark, ElasticNet effective L1 weight 5.9e-4 ≈ Lasso optimum, softened E002 fold-2 cause, Lasso sharp-optimum mechanism). Documentation measurements: Lasso E056 refit on 1,241 development rows keeps 115 of 352 coefficients (82 of 299 one-hot columns); largest coefficient `RoofMatl_ClyTile` −0.60 from a single house (memorisation lesson); exact collinearity in the base data (GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF; TotalBsmtSF = sum of the three basement areas; numeric design matrix rank 34, condition number ~7e15); Ridge E055 keeps 351 of 352. All four drafts by A (1, 2a, 2b) and C (3) are complete and committed; remaining study-guide chapter: boosting/Optuna (Agent B, not started).
