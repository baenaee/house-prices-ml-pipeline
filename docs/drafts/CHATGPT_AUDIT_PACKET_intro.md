# Audit packet for an external reviewer

This document is addressed to an independent reviewer (a person or another AI model) who is asked to find what is wrong with this project. It states what is claimed, where the evidence is, how each claim can be checked, and where the authors themselves think the weak points are. It was assembled by the research director; the "Claims to verify" section was written by the project's validation auditor, who had read access only to the pipeline code.

## What you are asked to do

1. Read the summary of claims below and the files listed under "Reading order".
2. For each claim in "Claims to verify", judge whether the evidence supports it. Where you cannot run code, read the named file and check that the logic matches the claim; where the owner can run the command for you, the expected output is given.
3. Look for methodological errors the internal audit may have missed: leakage paths, misuse of the holdout, selection bias that is not accounted for, statistical claims stronger than the data, numbers in the documents that do not match the registry.
4. Challenge the final decision (single CatBoost instead of the Lasso + CatBoost average) and the wording used to report it.
5. Report findings with a severity, the file and line, and what you would need to see to consider it resolved.

## The project in ten lines

- Task: Kaggle *House Prices: Advanced Regression Techniques*; metric RMSE on `log1p(SalePrice)`.
- 1,460 training rows; 219 (15 %, stratified by price decile, seed 42) set aside once as a holdout; 1,241 development rows.
- Selection: `KFold(5, shuffle, seed 42)` on development rows, all fitted preprocessing inside the fold.
- 83 registered experiments (`experiments/registry.jsonl`), each reproducible by `python -m houseprices.evaluate --experiment <id>`.
- Finalists re-scored on `RepeatedKFold(5×3, seed 7)`.
- Frozen model (ruling D22): single CatBoost, registry id E066, hash `be58905cb40a`.
- Claimed expected error: about 0.12 (fresh-fold CV 0.1217, fold sd 0.013).
- Holdout, scored once after the freeze: 0.10645 (95 % bootstrap interval 0.092–0.121).
- Best alternative: equal-weight average of Lasso and CatBoost, 0.1207 on fresh folds, 0.0010 better, about one third of a standard error; not chosen.
- Submission file written and validated; not submitted; no leaderboard score is claimed.

## Reading order

| file | what it gives you |
|---|---|
| `README.md` | overview, layout, commands |
| `docs/RESEARCH_REPORT.md` | all results and the limitations |
| `docs/EXPERIMENT_HISTORY.md` | chronology; rulings D1–D22 with the evidence for each; the failures |
| `docs/VALIDATION_AUDIT.md` | the internal audit: findings F1–F36, sign-offs, the pre-registered decision rule (§9) and holdout range (§9.9), the final round (§10) |
| `src/houseprices/validation.py`, `data.py`, `models.py`, `preprocess.py`, `ensemble.py` | the code that carries the methodology |
| `experiments/registry.jsonl`, `experiments/holdout_usage.log`, `experiments/finalists/*.csv` | the raw records |
| `git log` | the order of events: freeze `95c3e86` → approval `5beae61` → holdout `701b06f` → submission `b249ba9` |

## Where the authors think the project is weakest

1. **The final choice is a tie.** The pre-registered rule favoured the simpler model on ties. A reviewer may reasonably argue that an equal-weight average has no fitted parameters and that "best point estimate on both sets of folds" should have won. The documents report both numbers.
2. **Fresh folds are not fresh data.** The seed-7 folds re-partition the same 1,241 rows used for every selection decision. They remove the fold-specific part of the optimism, not any overfitting to the development sample as a whole. Only the holdout addresses that, and it has 219 rows.
3. **The holdout is favourable.** It contains no house above 4,000 sq ft; the fresh-fold CV estimate lies just above its 95 % interval. The documents quote 0.12, not 0.106, as expected performance. Check that no document slips into presenting 0.106 as the headline.
4. **Standard errors.** The finalists runner computes sd/√15 on 15 correlated folds (finding F30). The documents use sd/√5 and mention the Nadeau–Bengio correction. Neither is exact.
5. **Search-space edges.** Both Stage 2 optima sit on the 2,000-tree bound.
6. **The internal auditor is not independent in the strong sense.** It is an agent of the same team, with read-only access and its own tests; you are the independent check.
7. **A re-prediction on the holdout inside the scoring script** (finding F35): one scoring call, then the same fitted model predicted again to save residuals. No decision used it; judge whether you accept that.
8. **Test-only anomalies** (Ids 2593, 2550, 1877, 2121) cannot be validated.
9. **Partial experiment.** The CatBoost native-categorical study stopped at 5 of 25 trials.

## Questions we would like answered

- Is there any path by which holdout rows, or statistics computed from them, could have influenced a model, a hyper-parameter or a decision before commit `701b06f`?
- Is the nested protocol for blending and stacking correct as implemented in `ensemble.py` (inner out-of-fold predictions inside `fit`)?
- Is the outlier-policy experiment (filters on training folds only) correctly implemented in `validation.iter_folds`, and is the conclusion (keep all rows) justified by E049–E054?
- Are the statements about selection optimism (0.003–0.004) and about the holdout (consistency check, not headline) worded no more strongly than the evidence allows?
- Would you have frozen the average instead? What evidence would change your answer?

---

