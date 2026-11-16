# Pending document fixes (auditor's F4 consistency pass, 2026-11-17)

Saved at the usage-limit stop. NONE of these has been applied. No score, difference, standard error, count, price, commit id or experiment id was found wrong. After applying 1–3 (and 4–11), mark claim F4 closed in `docs/CHATGPT_AUDIT_PACKET.md`, append this pass to `docs/VALIDATION_AUDIT.md` as §10.9, rebuild the study guide (`docs/drafts/build_study_guide.py`; edit the drafts, not `STUDY_GUIDE.md`) and delete this file.

## Wording stronger than the evidence (fix first)

1. `README.md` results table and `docs/RESEARCH_REPORT.md` §1: "folds not used for any choice". The seed-7 folds were used once, under the pre-registered rule, to choose among eight finalists. Use: "folds not used for tuning or for generating candidates; used once, under a pre-registered rule, to choose among eight finalists".
2. `docs/RESEARCH_REPORT.md` §2 table: finalist folds "disjoint from the selection partition". Correct: "no fold identical to a selection fold" (validation overlap up to 25 %, 20 % expected by chance; audit F20).
3. `docs/EXECUTIVE_SUMMARY.md` ("36 findings ... each with a test") and study guide part 6 §6 ("each audit finding became a test"): only the bugs have tests; Info/Note findings (F22, F23, F27, F28, F29, F31, F32, F35, F36) do not. Use "each bug with a test".

## Small numeric or description errors

4. `RESEARCH_REPORT.md` §4.5: LightGBM minutes (both stages) should be 4.9, not 4.8 (1.7 + 3.2; total 72.0).
5. Study guide part 6 intro: `src/houseprices` is 2,439 lines (not "about 2,600"); suite is 217 tests (not "roughly 215").
6. Study guide part 6 §5: `run_outlier_policy.py` and `run_linear_family.py` take `--dry-run-rows N`; the ablation, LotFrontage, ensemble, Optuna and interpretability runners have `--dry-run`.
7. Study guide part 4 §1: says "eight candidates" but the table has seven rows; add E061/E081 (0.1181, 0.1215, +0.0034) or say seven.
8. Study guide part 4 §2 and self-check answer 2: "between a quarter and a half of one standard error" should be "between a quarter and a little over half" (0.26, 0.33, 0.57 for the three standard errors).
9. Study guide part 7 (D16 row and "Ridge as a finalist"): "by 0.005 in every fold" → per-fold gaps are 0.003–0.007, mean 0.0048; say "in every fold, by 0.005 on average".
10. `RESEARCH_REPORT.md` §1: "removes about 70 % of the mean baseline's error on the same folds": say "on the selection folds" (70.4 %) or drop "on the same folds".
11. "Seven audit rounds" (`README.md`, `EXECUTIVE_SUMMARY.md`): the audit file has rounds 1, 2, 2b, 3, 4, 5, 6 and Final; say "seven rounds plus a final round".
