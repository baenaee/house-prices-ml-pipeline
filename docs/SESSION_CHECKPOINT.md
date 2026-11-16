# Session checkpoint — 2026-11-17 (session V4: campaign COMPLETE)

Purpose: state of the project at the end of the research campaign. Earlier checkpoints are superseded. Nothing is running and no task of the approved plan is open; the only remaining actions are the owner's (section 5).

## 1. Repository state

- Git: `C:\Program Files\Git\cmd\git.exe`. Final commit: see `git log -1`. Key commits of this session, oldest first: `fb3c03d` boosting chapter draft + interpretability runner · `044a834` Stage 2 tuning · `c9168a0` ensembles on final bases · `4f2db9b` finalists · `23f56ac` FREEZE (ruling D22) · `264f43f` auditor approval + holdout script · `c6943b1` HOLDOUT SCORED ONCE · `385a542` final fit + submission file · `afa49de` report, study guide, notebooks 03–05 · `31a13db` final audit, summary, audit packet · closing commit.
- Originals byte-identical to `fedba71` (SHA-256 re-verified by the auditor in the final round: notebook 1BB32B57…, train 1E18ADDF…, test 8FDD3D82…).
- Environment: conda env `houseprices`, python `C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe` (plain `python` not on PATH).
- Tests: `& "<env python>" -m pytest tests` → **217 passed, 0 failed** (do not add `-q`).
- Registry: `experiments/registry.jsonl`, **83 experiments, E001–E083**, ids and hashes unique, means equal fold means (final audit §10.4).
- Git-ignored local files worth keeping: `experiments/tuning/optuna_stage1.db` (+ `optuna_stage1.pre_stage2.db`), `outputs/models/E066_development.joblib`, `outputs/models/E066_all.joblib` and their JSON sidecars.

## 2. Final results

| item | value | evidence |
|---|---|---|
| Frozen final procedure (D22) | single CatBoost, **E066**, hash `be58905cb40a` | `docs/EXPERIMENT_HISTORY.md` D22; freeze commit `23f56ac` |
| Selection-fold CV (seed 42) | 0.11807 | `evaluate --experiment E066` |
| Fresh-fold CV (RepeatedKFold 5x3, seed 7) | **0.12166**, fold sd 0.0131 | E076 |
| Best alternative: average Lasso + CatBoost (E068) on fresh folds | 0.12066 (−0.0010 vs E066, SE about 0.003) | E077 |
| Holdout, 219 rows, ONE scoring call | **0.10645**, 95 % bootstrap interval 0.092–0.121 | `experiments/holdout_usage.log`, audit §10.2 |
| Submission file | `outputs/submissions/E066_all.csv`, 1,459 rows, validated, **not submitted** | audit §10.3 (independent rebuild identical to 3e-16) |
| Tuning time used | 72 of 180 minutes | audit §9.1 |

The number to quote as expected performance is about 0.12 (fresh-fold CV). The holdout is a consistency check on the favourable side (no house above 4,000 sq ft in it).

## 3. Completed blocks

Phase 0, A, B; C1–C8 (ablations, outlier policy, LotFrontage, linear grids, Stage 1 and Stage 2 boosting, ensembles on Stage 1 and final bases, finalists, freeze, holdout); E1 interpretability (`experiments/interpretability/run_interpretability.py`, `outputs/figures/`, `outputs/interpretability/`); F1 train/predict/submission; F2 final audit (VALIDATION_AUDIT §9–§10, findings F1–F36, all closed, accepted or noted); D1 documentation: `README.md`, `docs/RESEARCH_REPORT.md`, `docs/STUDY_GUIDE.md` (about 34,000 words, built from `docs/drafts/` by `docs/drafts/build_study_guide.py`), `docs/EXPERIMENT_HISTORY.md` (D1–D22), `docs/VALIDATION_AUDIT.md`, `docs/CHATGPT_AUDIT_PACKET.md`, `docs/EXECUTIVE_SUMMARY.md`; notebooks 01–05 with executed copies.

## 4. Standing rules that still apply to any future work

- The holdout is spent. `validation.score_holdout_once` raises on a second call; any further model change makes 0.10645 a biased estimate for the new model and must be reported as such.
- Never regenerate `experiments/holdout_ids.json`; never modify the root originals; one registry writer at a time (F27).
- `docs/STUDY_GUIDE.md` is generated: edit the drafts and re-run the build script.
- No Kaggle submission without the owner's decision.

## 5. Open items (owner's decisions, not blockers)

1. Whether to upload `outputs/submissions/E066_all.csv` to Kaggle. If you do, record the public score in `docs/EXPERIMENT_HISTORY.md` and do not tune against it.
2. Whether to send `docs/CHATGPT_AUDIT_PACKET.md` to the external reviewer, and which findings to act on.
3. Optional follow-ups listed as limitations in the report: wider tree-count search; finishing the CatBoost-native study (about 2 h); a segment model or features for very large and very cheap houses; a time-aware validation split.
