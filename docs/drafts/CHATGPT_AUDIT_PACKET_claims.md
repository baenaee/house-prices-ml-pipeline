DRAFT — status: complete (final round 2026-11-17: claims A-E verified, new claims C7-C9 and F1-F4 added with real results; remaining: re-run the suite count after the director updates the stale guard test, finding F34)

# Claims to verify

Draft section for `docs/CHATGPT_AUDIT_PACKET.md`. Status date: 2026-11-17. Repository state: 83 registry records, 217 tests in 9 test files (216 pass until the stale test named in finding F34 is updated; 217 afterwards), HEAD after commit `385a542`.

## How to use this section

The reviewer can read the repository but cannot run code. The owner can. For every claim the table gives the evidence, one command and the output that confirms it. The "look for" line says how the claim could be false. A reviewer who cannot run the command should read the named file and check that the logic matches the claim.

All commands run from the project root in PowerShell. Set the interpreter once:

```powershell
$PY = "C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe"
```

Test counts below are `passed` lines from pytest. The probes in `tests/test_audit_round2.py` were written by the validation auditor and are independent of the code owners' own tests. Nothing in this section was run on Kaggle, and no submission has been made.

Section F now holds real results (finalists, holdout, submission). Nothing was submitted to Kaggle and no leaderboard score is claimed.

---

## A. Data cleaning

### A1. The 17 cleaning rules reproduce the notebook's verification counts

- **Evidence:** `tests/test_cleaning.py`, dictionary `EXPECTED`; the notebook counts are tabulated in `docs/AUDIT_2026-11-11.md`.
- **Command:** `& $PY -m pytest tests/test_cleaning.py`
- **Expected:** `30 passed`.
- **Counts asserted (train, test):**

| Rule | Train | Test | Rule | Train | Test |
|---|---|---|---|---|---|
| V01 | 864 | 878 | A01 | 1453 | 1453 |
| V02 | 8 | 16 | A02 | 690 | 730 |
| V03 | 7 | 3 | A03 | 1406 | 1407 |
| B01 | 37 | 41 | A04 | 1369 | 1352 |
| B02 | 2 | 7 | A05 | 1179 | 1169 |
| B03 | 0 | 1 | A06 pool | 0 | 3 |
| B04 | 0 | 1 | A06 misc | 0 | 1 |
| G01 | 81 | 76 | C01 entries | 1 | 12 |
| G02 | 0 | 2 | C01 houses | 1 | 9 |
| G03 | 0 | 2 | | | |

- **Look for:** a count copied from the wrong notebook output, or a rule whose count matches while the filled values are wrong. The tests also assert that no object column has NaN after cleaning, that deferred numeric NaN remain (LotFrontage, GarageYrBlt, MasVnrArea), and that cleaning twice changes nothing.

### A2. Train and test receive the same cleaning, row by row, and are never combined

- **Evidence:** `cleaning.clean`, `cleaning.clean_train_test`; probes `test_clean_is_row_wise_so_train_and_test_cannot_influence_each_other`, `test_cleaning_never_reads_target_or_fits_statistics`, `test_train_and_test_use_same_function_and_same_columns`, `test_every_feature_group_is_row_wise`.
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "never_reads or row_wise or same_function"`
- **Expected:** `4 passed`.
- **What the probes prove:** cleaning 300 random rows gives exactly the same rows as cleaning the full table, so no rule uses information from other rows. A syntax-tree scan finds no reference to SalePrice, no median, mean, mode, group statistic or `fit` call in `cleaning.py` or `features.py`, and every `fillna` argument is a constant.
- **Look for:** any statistic computed from a whole column in `cleaning.py`; any place where `train` and `test` frames are concatenated (a search for `concat` and `merge` in that file should find nothing).

### A3. The reconstructed rule C01 is labelled as a reconstruction

- **Evidence:** docstring of `cleaning.rule_c01`; comment on `config.C01_COLS`. The original notebook cell is lost in every available copy.
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "c01_is_labelled"` then `Select-String -Path src\houseprices\cleaning.py -Pattern "RECONSTRUCTION"`
- **Expected:** `1 passed`; one match in the `rule_c01` docstring saying "NOT recovered code".
- **Look for:** C01 described anywhere as "recovered" or "restored". Its behaviour is checked only against the saved verification output (1 entry and 1 house in train, 12 and 9 in test) and the PDF log, not against original code.

### A4. B04 is a condition, not a hard-coded Id

- **Evidence:** `cleaning.basement_fully_unrecorded`, `cleaning.rule_b04`; probe `test_b04_is_a_condition_selecting_only_test_2121`.
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "b04"`
- **Expected:** `1 passed`. The condition (all 5 basement categoricals and all 6 basement numerics missing) selects no training row and exactly test Id 2121; the number 2121 appears only in docstrings.
- **Look for:** a literal Id in code, or a condition that also catches training rows.

### A5. The data entry point re-attaches Id (finding F9) and splits before building features

- **Evidence:** `data._with_id`, `data.load_development_data`; `tests/test_data.py`; probes `test_base_group_excludes_id_by_design_and_data_reattaches_it`, `test_holdout_split_then_features_equals_features_then_split`.
- **Command:** `& $PY -m pytest tests/test_data.py` and `& $PY -m pytest tests/test_audit_round2.py -k "reattaches_it or holdout_split_then_features or y_log_aligned"`
- **Expected:** `4 passed`; `3 passed`.
- **History:** the feature group `base` drops Id, so the outlier filter (which selects Ids 524 and 1299) raised `KeyError` on the feature table. The director fixed it; the auditor verified it. Models drop Id before preprocessing, so it never reaches an estimator (the per-fold probe checks that Id is absent from the fitted feature names).
- **Look for:** Id used as a numeric feature; target values aligned to rows by position rather than by Id (the probe compares `y_log` to `log1p(SalePrice)` by Id).

### A6. Test-only anomalies are not trained on (rulings D18)

- **Evidence:** `cleaning.add_anomaly_flags`; the three flags are all zero in the training data, so the group `anomaly_flags` cannot change a CV score. Test Ids: 2593 (garage year 2207), 2550 (sold before built), 1877 (remodelled before built), 2121 (no basement record).
- **Command:** `& $PY -m pytest tests/test_features.py -k "anomaly"`
- **Expected:** `2 passed`; the flag sums on test are 2, 1, 1 (houses 2593, 2550, 1877; Id 2550 also has an implausible garage year).
- **Look for:** the model's behaviour on these four houses is untested against truth, because no training row resembles them. See the residual-risk list.

---

## B. Preprocessing and fold safety

### B1. Everything that learns from data is fitted inside the training fold

Covers `GroupMedianImputer` (LotFrontage by Neighborhood), `SimpleImputer`, `OneHotEncoder`, `RobustScaler`/`StandardScaler`, `SkewLog1pTransformer`, `KNNLotFrontageImputer` and `YearRangeClipper`.

- **Evidence:** `src/houseprices/preprocess.py`, `src/houseprices/models.py` (`LazyPreprocessor`, `NumericMedianImputer`); probes `test_group_median_imputer_learns_only_in_fit_from_given_rows`, `test_lazy_preprocessor_refits_per_fold_with_training_medians`, `test_skew_pipeline_in_cv_selects_columns_per_fold_and_is_row_independent`, `test_knn_transform_is_row_independent_and_ignores_new_lotfrontage_values`, `test_year_clipper_learns_only_from_fit_rows_and_clips_unseen_years`, `test_make_preprocessor_reads_only_names_and_dtypes`.
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "group_median or skew or knn or year_clipper or lazy_preprocessor_refits or numeric_median or reads_only_names"`
- **Expected:** `13 passed`.
- **What the probes prove:** under a 5-fold run with returned estimators, each fold's fitted numeric medians equal the medians of that fold's training rows, differ between folds, and the template estimator stays unfitted. Skewed columns are selected per fold. KNN donors are training rows only: changing LotFrontage in other validation rows does not move an imputed value. Transforming one row gives the same result as transforming it inside a batch.
- **Look for:** a transformer that is fitted on the whole development set before `cross_validate`; `fit_transform` on the full table in any `experiments/` script (the scripts call `_common.run_cv`, which only passes raw feature frames).

### B2. The preprocessor reads only column names and dtypes when it is built

- **Evidence:** `preprocess.make_preprocessor`, `models.LazyPreprocessor.fit`; probe `test_make_preprocessor_reads_only_names_and_dtypes` (identical structure for the full frame, a shuffled copy and an empty frame).
- **Command:** included in B1.
- **Look for:** a builder in `models.py` that receives data (builders take keyword arguments only).

### B3. Optional steps are off in the baselines

- **Evidence:** `make_preprocessor` defaults (`skew_log1p=False`, `clip_years=False`, `lotfrontage="neighborhood_median"`); probe `test_default_preprocessor_has_no_optional_components`.
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "default_preprocessor_has_no_optional"`
- **Expected:** `1 passed`. Linear models enable `skew_log1p` explicitly (ruling D12), recorded inside each experiment's `builder_kwargs`; trees never do.

---

## C. Validation protocol

### C1. The metric is RMSE on log1p(SalePrice), which equals the competition's RMSLE

- **Evidence:** `validation.to_log_target`, `rmse_log`, `rmsle_dollars`; `tests/test_validation.py`.
- **Command:** `& $PY -m pytest tests/test_validation.py`
- **Expected:** `22 passed` (hand computation: errors 0.1, -0.2, 0.3 give 0.2160247; dollar-scale RMSLE equals the log-scale value).
- **Look for:** a score computed on dollars, or predictions transformed with `exp` instead of `expm1`.

### C2. Outlier-style row filters touch only the training part of a fold, and validation folds stay complete (rulings D3, D15)

- **Evidence:** `validation.iter_folds` and `cross_validate`; registry ids E049 to E054 versus their keep_all counterparts E041 (Ridge), E042 (ElasticNet), E023 (XGBoost); `docs/EXPERIMENT_HISTORY.md` section "outlier policy".
- **Command:**

```powershell
@'
import numpy as np
from houseprices import data, validation
d = data.load_development_data("base")
ids = d.X["Id"].to_numpy()
for k, (tr, va) in enumerate(validation.make_cv(5, 1, 42).split(d.X)):
    print("validation fold index", k, "contains outlier Ids", [i for i in (524, 1299, 692, 1183) if i in set(ids[va])])
'@ | & $PY -
```

- **Expected:** fold index 1 contains 1299, 692 and 1183; fold index 2 contains 524; the others none (indices start at 0).
- **The story:** the Ridge baseline E003 scores 0.185 on fold index 1, the hard fold for every linear model, because the very expensive and the very large houses are there. Dropping Ids 524 and 1299 from the training part only (E049) moves that fold from 0.161 to 0.184 while the other folds hardly change (for example 0.113 to 0.107), so the mean gets worse (0.1304 to 0.1348). The model trained without extreme houses is still judged on them. Removal "helps" in public notebooks only when the same rows are also removed from the validation side. Ruling D15 therefore keeps every row.
- **Look for:** a filter applied before the split; a validation fold with fewer than its natural rows. Probe: `& $PY -m pytest tests/test_validation.py -k "filter"` (expected `4 passed`).

### C3. Folds are seeded and deterministic, and the registry-based `evaluate` command reproduces registered scores

- **Evidence:** `validation.make_cv`, `evaluate.run_registered`; registry ids E003, E004, E042, E056, E060.
- **Commands (run one at a time; E060 takes about 20 seconds):**

```powershell
foreach ($e in "E003","E004","E042","E056","E060") { & $PY -m houseprices.evaluate --experiment $e | Select-String "reproduces_registered_scores|registered_mean|reproduced_mean" }
```

- **Expected:** each experiment prints `"reproduces_registered_scores": true`. Registered means: E003 0.140883 (Ridge alpha 10), E004 0.148681 (XGBoost defaults), E042 0.125622 (ElasticNet, feature set all, skew), E056 0.12545 (Lasso best of grid), E060 0.119457 (CatBoost best of 25 trials). Fold scores agree to 1e-5.
- **Look for:** reproduction that merely reads the stored numbers. `evaluate` rebuilds the estimator from the stored builder name and keyword arguments, reloads the data through the same entry point, refits per fold and compares (`src/houseprices/evaluate.py`).

### C4. The registry is internally consistent

- **Evidence:** `experiments/registry.jsonl` (append-only, one JSON record per experiment).
- **Command:**

```powershell
@'
import json, collections, numpy as np
recs = [json.loads(l) for l in open("experiments/registry.jsonl", encoding="utf-8") if l.strip()]
ids = collections.Counter(r["id"] for r in recs)
hashes = collections.Counter(r["config_hash"] for r in recs)
print("records", len(recs))
print("duplicate ids", [i for i, n in ids.items() if n > 1])
print("duplicate hashes", [h for h, n in hashes.items() if n > 1])
print("mean mismatches", [r["id"] for r in recs if abs(np.mean(r["fold_scores"]) - r["cv_mean"]) > 2e-6])
print("fold-count mismatches", [r["id"] for r in recs if len(r["fold_scores"]) != r["cv_n_splits"] * r["cv_n_repeats"]])
'@ | & $PY -
```

- **Expected (at 83 records):** all four lists are empty `[]`. The eight finalist records E076-E083 repeat their source configuration with seed 7; the seed is part of the config hash, so they do not appear as duplicates. The record count grows as experiments are added.
- **Look for:** a stored `cv_mean` that is not the mean of its fold scores; an experiment registered twice under different ids.

### C5. Selection on five folds is optimistic by about 0.002 to 0.003, and finalists are re-scored on a different partition (rulings D17)

- **Evidence:** `docs/VALIDATION_AUDIT.md` sections 7.3 and 7.4; `config.CV_SEED_FINALISTS = 7`; `evaluate --seed`; `experiments/finalists/run_finalists.py` (refuses seed 42).
- **Command 1 (Lasso E056 on seed-7 folds, writes nothing):** `& $PY -m houseprices.evaluate --experiment E056 --repeats 3 --seed 7 | Select-String "registered_mean|reproduced_mean|reproduces"`
- **Expected 1:** registered mean 0.12545, reproduced mean 0.12816 (optimism +0.0027), and `n/a (different repeats or seed)` for reproduction.
- **Command 2 (Ridge grid, about 2 minutes, writes nothing):**

```powershell
@'
import numpy as np, warnings
from houseprices import data, models, validation
warnings.filterwarnings("ignore")
d = data.load_development_data("base")
alphas = np.logspace(-1, 2.5, 15)
S = np.array([[validation.cross_validate(models.build_ridge(alpha=float(a), preprocessor_kwargs={"skew_log1p": True}), d.X, d.y_log, validation.make_cv(5, 1, seed)).mean for a in alphas] for seed in (42, 1, 2, 3, 4, 5)])
j = S[0].argmin()
print("selected alpha", round(alphas[j], 2), "selection-fold score", round(S[0, j], 4), "mean on 5 fresh partitions", round(S[1:, j].mean(), 4))
print("seed-to-seed std of the CV mean", round(S.std(axis=0, ddof=1).mean(), 4))
'@ | & $PY -
```

- **Expected 2:** alpha 10; 0.1319 on the selection folds against about 0.1340 on fresh partitions (+0.0021); seed-to-seed standard deviation about 0.0026.
- **Look for:** a claim that a tuned model beats another by less than about 0.003 (within selection noise). Finding F20 recorded that the first repeat of `RepeatedKFold(random_state=42)` is identical to the selection folds; the check `& $PY -m pytest tests/test_audit_round2.py -k "finalist_seed_folds"` (expected `1 passed`) confirms that seed 7 shares no fold with seed 42 (largest validation overlap 25%, chance level 20%).

### C6. Nested stacking never scores the meta-learner on rows it was fitted on

- **Evidence:** `ensemble.NestedStackingRegressor`, `BlendingRegressor`, `inner_oof_matrix`; `tests/test_ensemble.py`; auditor probes in `tests/test_audit_round2.py` (round 3 block).
- **Command:** `& $PY -m pytest tests/test_ensemble.py` then `& $PY -m pytest tests/test_audit_round2.py -k "nested or leaky or blend_weights or stack_registers or optimistic"`
- **Expected:** `15 passed`; `7 passed`.
- **What the probes prove:** inside each outer fold, a recording base model shows six fits (five inner, one refit), all on outer-training rows and none touching the outer validation rows; the meta-learner's training matrix equals the inner out-of-fold predictions recomputed from outer-training rows alone; the bases are refitted on the whole outer-training partition; blend weights are learned inside `fit`. A deliberately leaky stack (meta-learner trained on in-sample base predictions) is more than three times worse than the honest one when one base memorises its training rows, so the probe is sensitive.
- **Look for:** weights or meta-learner fitted on out-of-fold predictions from the whole development set and then scored on those rows. The "optimistic" helper functions return plain floats and cannot be registered through the shared runner (probe `test_optimistic_diagnostics_cannot_flow_through_run_and_register`); a hand-built registry record could still carry one (residual risk below).

### C7. Stage 2 tuning: the registered E066 and E067 are the best trials of their 75-trial studies, scored on development rows only (audit 9.1)

- **Evidence:** `experiments/tuning/run_optuna_stage1.py` (objective calls `_common.run_cv` only), the two `*_trials.csv` files, registry E066 and E067, `docs/VALIDATION_AUDIT.md` section 9.1. The studies were resumed from 25 to 75 trials; the `--stage 2` flag only labels the record.
- **Command:**

```powershell
@'
import json, pandas as pd
fs = "base+areas+baths+ages+quality_interactions+ordinal_quality"
reg = {json.loads(l)["id"]: json.loads(l) for l in open("experiments/registry.jsonl", encoding="utf-8") if l.strip()}
for name, eid in (("catboost", "E066"), ("lightgbm", "E067")):
    df = pd.read_csv(f"experiments/tuning/stage1_{name}_{fs}_trials.csv")
    b = df.loc[df.value.idxmin()]
    print(eid, "trials", len(df), "numbers unique", df.number.is_unique, "best trial", int(b.number), "value %.6f" % b.value,
          "params equal registry", json.loads(b.user_attrs_builder_kwargs) == reg[eid]["builder_kwargs"])
'@ | & $PY -
& $PY -m houseprices.evaluate --experiment E066
```

- **Expected:** `E066 trials 75 numbers unique True best trial 43 value 0.118072 params equal registry True`; `E067 trials 75 numbers unique True best trial 35 value 0.121733 params equal registry True`; `evaluate` prints `"reproduces_registered_scores": true`. Tuning time: 10.8 + 28.0 minutes (Stage 1) plus 30.0 + 3.2 (Stage 2) = 72.0 of the 180 allowed.
- **Look for:** a best trial whose parameters differ from the registered ones; duplicate trial numbers after the resume; the registered record carrying Stage 1 wording (finding F29, a text issue only); parameters at the edge of the search box (2,000 iterations), which is a stated limitation.

### C8. The ensembles E068-E075 are scored by the outer CV only; their paired differences are arithmetic on registered fold scores (audit 9.2)

- **Evidence:** `src/houseprices/ensemble.py` (weights and meta-learner fitted inside `fit` from inner out-of-fold predictions), `experiments/ensemble/run_ensembles.py`, `experiments/ensemble/ensembles_20261116_234858.csv`.
- **Command:**

```powershell
@'
import json, numpy as np, pandas as pd
reg = {json.loads(l)["id"]: json.loads(l) for l in open("experiments/registry.jsonl", encoding="utf-8") if l.strip()}
c = pd.read_csv("experiments/ensemble/ensembles_20261116_234858.csv").set_index("id")
for i in c.index:
    d = np.array(reg[i]["fold_scores"]) - np.array(reg[c.loc[i, "best_member"]]["fold_scores"])
    print(i, "delta %.6f vs csv %.6f  se %.6f vs csv %.6f" % (d.mean(), c.loc[i, "delta"], d.std(ddof=1) / 5 ** 0.5, c.loc[i, "paired_se"]))
'@ | & $PY -
```

- **Expected:** eight lines in which the recomputed and CSV values agree to six decimals (for example E068 -0.000728 / SE 0.001943). `evaluate --experiment E068` and `E072` report `reproduces_registered_scores` true.
- **Look for:** a weight or meta-learner fitted on rows later scored; a diversity pass whose output enters a registered number (it does not); a paired difference against a member that is not the best one on the same folds.

### C9. The final model was chosen by a rule written before the finalist table was read (audit 9.3, 9.6, ruling D22)

- **Evidence:** `docs/VALIDATION_AUDIT.md` section 9.3 (rule, saved in the file before the finalist records were opened) and 9.6 (application); registry records E076-E083 (seed 7, 5x3); `docs/EXPERIMENT_HISTORY.md` ruling D22.
- **Command:** `& $PY -m houseprices.evaluate --experiment E076` (RepeatedKFold 5x3, seed 7; about 90 seconds) and the recomputation in F1 below.
- **Expected:** `reproduces_registered_scores` true (0.12166 vs 0.121664). The rule keeps single CatBoost E066 unless an ensemble has a paired difference of at most -0.0010, at most -1 adjusted SE, and wins at least 10 of 15 folds with all three repeat means negative. The best ensemble (E068) had -0.0010, adjusted SE 0.0031 and 9 of 15 folds: a tie, resolved by parsimony.
- **Look for:** a rule written after the table (the audit file was edited before the finalist registry lines existed; the repository does not time-stamp the edit, so this rests on the auditor's report and the registry line count it names, 75 then 77); a ranking by seed-42 scores.

---

## D. Holdout and training script

### D1. The holdout was carved once, has 219 ids, and was scored exactly once, after the procedure was frozen

- **Evidence:** `experiments/holdout_ids.json` (single commit `582bf86`); `experiments/holdout_usage.log` (one line); commits `23f56ac` (freeze, ruling D22), `264f43f` (scoring script and auditor approval with the pre-registered range, `docs/VALIDATION_AUDIT.md` 9.9), `c6943b1` (the scoring).
- **Command:**

```powershell
Get-Content experiments\holdout_usage.log
git log --oneline -- experiments/holdout_ids.json
git log --format="%h %ad %s" --date=iso 23f56ac..c6943b1
```

- **Expected:** exactly one JSON line: `frozen_experiment_id` E066, `n_holdout` 219, `rmse_log` 0.10645, `forced` false, `n_previous_uses` 0; one commit for `holdout_ids.json`; the freeze commit precedes the scoring commit.
- **Look for:** more than one log line, `forced` true, a log time earlier than the freeze commit, a regenerated `holdout_ids.json`.

### D2. The holdout can be scored once, only with the registered holdout rows, and a forced second use is marked

- **Evidence:** `validation.score_holdout_once`, `_assert_is_holdout_frame` (finding F21, fixed); `tests/test_holdout_guard.py`.
- **Command:** `& $PY -m pytest tests/test_holdout_guard.py`
- **Expected:** `5 passed`. A second call raises `HoldoutAlreadyUsedError` and logs nothing; `force=True` logs `"forced": true`; a frame whose Id set differs from the stored ids is refused without consuming the use; a non-finite prediction does not consume it.
- **Look for:** a direct call to `predict` on `X_holdout` outside this function (not logged); the report must state which logged call is the official one.

### D3. Model-selection code never touches the holdout

- **Evidence:** probes `test_run_cv_never_touches_holdout_frames`, `test_round4_scripts_never_reference_holdout_or_bypass_run_cv`, `test_ablation_scripts_never_touch_holdout_and_use_only_common_entry_points`, `test_train_and_predict_never_reference_holdout_frames`.
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "holdout_frames or bypass_run_cv or ablation_scripts_never or never_reference_holdout"`
- **Expected:** `4 passed`.

### D4. Fitting on all training rows was refused until the holdout had been scored (finding F24, fixed); the final artefacts were fitted in the right order

- **Evidence:** `train.fit_registered`; sidecar keys `includes_holdout_rows`, `holdout_scored_before_fit`; `outputs/models/E066_development.json` (rows development, 1,241 rows, `holdout_scored_before_fit` false, written 00:40:42, before the log line at 00:40:45) and `E066_all.json` (rows all, 1,460 rows, `holdout_scored_before_fit` true, 00:41:04).
- **Command:** `& $PY -m pytest tests/test_audit_round2.py -k "f24"` and read the two sidecars. (Do NOT run `train --rows all` to test the refusal now: the holdout is scored, so the command would succeed and write a file.)
- **Expected:** the probe passes (refusal while the usage log is empty, using a temporary log); sidecar fields as listed; `git diff --stat 23f56ac HEAD -- src` is empty.
- **Look for:** an artefact named `*_all.joblib` older than the log line, or a `*_development` sidecar with `includes_holdout_rows` true.

### D5. `predict.py` checks its output before writing a submission

- **Evidence:** `predict.predict_test`: log predictions finite, prices finite and positive (F25, fixed), Ids and order identical to `sample_submission.csv`, columns `Id, SalePrice`. Nothing talks to Kaggle.
- **Command:** `& $PY -m pytest tests/test_pipeline_end_to_end.py`
- **Expected:** `4 passed`.

---

## E. Originals

### E1. The original files are unchanged

- **Evidence:** the SHA-256 manifest in `docs/AUDIT_2026-11-11.md` and the dated backup ZIP named there; commit `fedba71`.
- **Command:** `Get-FileHash -Algorithm SHA256 HousePricing.ipynb, train.csv, test.csv, data_description.txt, sample_submission.csv | Select-Object Hash, Path`
- **Expected hashes (start of each):** `HousePricing.ipynb` 1BB32B57, `train.csv` 1E18ADDF, `test.csv` 8FDD3D82, `data_description.txt` 356A4670, `sample_submission.csv` F19E9969 (checked on 2026-11-13 by the auditor; full hashes are in the manifest).
- **Look for:** a hash mismatch, or a cleaning step in `src/` that writes to a root CSV (`cleaning.py` only reads).

---

## F. Finalists, holdout, submission and final numbers

### F1. Finalist ranking on fresh folds (seed 7, RepeatedKFold 5x3), with the standard-error caveat (finding F30)

- **Evidence:** registry records E076-E083 (agent "director", tag `finalist`, `notes` "source experiment E0xx"), `experiments/finalists/finalists_20261117_003230.csv`.
- **Command:**

```powershell
@'
import json, numpy as np
reg = {json.loads(l)["id"]: json.loads(l) for l in open("experiments/registry.jsonl", encoding="utf-8") if l.strip()}
ref = np.array(reg["E076"]["fold_scores"])
for i in ["E077","E081","E078","E076","E079","E080","E083","E082"]:
    x = np.array(reg[i]["fold_scores"]); d = x - ref
    print(i, reg[i]["notes"], "seed", reg[i]["seed"], "n", len(x), "mean %.5f" % x.mean(), "d vs E066 %+.5f" % d.mean(),
          "SE runner %.5f adj %.5f" % (d.std(ddof=1) / 15 ** .5, d.std(ddof=1) / 5 ** .5))
'@ | & $PY -
```

- **Expected:** E077 (average) 0.12066, d -0.00101, adjusted SE 0.00305; E081 0.12147; E078 (stack) 0.12164; E076 (E066) 0.12166; E079 (blend) 0.12240; E080 (Stage 1 CatBoost) 0.12325; E083 (LightGBM) 0.12524; E082 (Lasso) 0.12816. All records seed 7 with 15 fold scores.
- **Caveat to state:** the 15 folds are three re-partitions of the same 1,241 rows, so the runner's SE (sd/sqrt 15) is too small; the adjusted SE (sd/sqrt 5) is about 0.003 and the data cannot separate E066 from the best ensemble. The choice of E066 is a parsimony decision.
- **Look for:** a finalist chosen from the seed-42 score alone; a claim that the ensemble is worse than CatBoost (it is a tie).

### F2. Single holdout score of the frozen model: 0.10645, 95% bootstrap interval 0.092 to 0.121

- **Evidence:** `experiments/holdout_usage.log` (one line, `forced` false); `experiments/finalists/holdout_predictions_E066.csv` (219 Ids equal to `holdout_ids.json`); `docs/VALIDATION_AUDIT.md` sections 9.9 and 10.
- **Command:**

```powershell
@'
import numpy as np, pandas as pd, json
h = pd.read_csv("experiments/finalists/holdout_predictions_E066.csv")
print("n", len(h), "rmse %.5f" % np.sqrt((h.residual_log ** 2).mean()),
      "ids equal", set(h.Id) == set(json.load(open("experiments/holdout_ids.json"))["holdout_ids"]))
rng = np.random.default_rng(0); sq = h.residual_log.values ** 2
b = [np.sqrt(rng.choice(sq, len(sq)).mean()) for _ in range(20000)]
print("95%% bootstrap interval %.4f - %.4f" % tuple(np.percentile(b, [2.5, 97.5])))
'@ | & $PY -
```

- **Expected:** `n 219 rmse 0.10645 ids equal True` and an interval of about 0.092 to 0.121. The pre-registered consistent range (written before scoring, audit 9.9) was 0.100 to 0.145.
- **Reading:** the score is an unbiased single draw for the model fitted on the 1,241 development rows. It is below the seed-7 CV estimate 0.1217 mainly because the holdout contains no house above 4,000 sq ft of living area: the four such houses (Ids 524, 692, 1183, 1299) are development rows and carry 8.5% of the squared out-of-fold error; without them the seed-42 pooled OOF RMSE falls from 0.1190 to 0.1140. Sampling noise (seed-7 fold scores range 0.103 to 0.150) and the fresh-fold level shift also play a part. The interval is the sampling error of the 219 houses for this fixed model and ignores the variance of the model fit. The holdout does not rank E066 against the average.
- **Look for:** `forced` true; a second log line; the score reported as the estimate for the model fitted on all rows (that model has seen the holdout and is not what was scored).

### F3. Final submission file: 1,459 rows, valid, not submitted

- **Evidence:** `outputs/submissions/E066_all.csv`, produced by `train --experiment E066 --rows all` (1,460 rows) and `predict` after the holdout was scored; sidecar `outputs/models/E066_all.json`.
- **Command:**

```powershell
@'
import pandas as pd, numpy as np
s = pd.read_csv("outputs/submissions/E066_all.csv"); ss = pd.read_csv("sample_submission.csv")
print(list(s.columns), len(s), (s.Id.values == ss.Id.values).all(), np.isfinite(s.SalePrice).all(), (s.SalePrice > 0).all())
print(s.SalePrice.describe().round(0).to_dict())
print(s.set_index("Id").loc[[1877, 2550, 2593]].round(0).to_dict())
'@ | & $PY -
```

- **Expected:** `['Id', 'SalePrice'] 1459 True True True`; minimum 41,901, median 156,611, maximum 524,842; Ids 1877, 2550, 2593 predicted 212,580, 268,438, 256,431. An independent refit of E066 on all 1,460 rows from the registry record reproduced every prediction to a relative difference below 1e-15 (audit section 10).
- **Look for:** an Id order different from the sample; implausible prices (the lowest, 41,901, is a 797 sq ft quality-3 house); any claim of a leaderboard score (none exists).

### F4. Final numbers in README, research report and study guide

- **Status:** closed 2026-11-17 after the document consistency pass (audit §10.9): no score, difference, SE, count or id was wrong; 11 wording and description corrections were applied (three over-strong claims qualified) and re-checked against the registry and finalist CSV; suite 217 passed. Every figure must equal a registry record, a finalist row, the usage log line or the audit sections above: E066 CV 0.1181 (seed 42, selection, optimistic) and 0.1217 (seed 7, fresh folds); holdout 0.10645 with interval 0.092 to 0.121; tuning 72 of 180 minutes.
- **Look for:** the seed-42 figure quoted as the performance estimate; the holdout quoted as proof that E066 beats the ensemble; a number without a registry id.

---

## How this project could still be wrong

1. **Selection optimism.** Registered best scores come from 136 grid points and up to 75 Optuna trials per booster judged on the same five folds. The measured optimism is about +0.002 to +0.003 for linear models and probably 0.002 to 0.005 for boosters (not measured). Differences between tuned models below about 0.003 are noise. Only the fresh-fold finalist check and the holdout bound this.
2. **The holdout is small.** With 219 rows its sampling error is of order 0.01 in RMSE. It cannot separate close models, and it is a single draw of houses.
3. **Test-only data gaps.** The three anomaly flags are zero in all training rows, so no model has seen a house like test Ids 2593, 2550, 1877 or 2121. Id 2550 (sold before it was built) is handled only by clipping house age at 0, and Id 2593 (garage year 2207) by the year clipper or by setting garage age to missing. Predictions for these four houses are unvalidated; they are printed by `predict.py` for a manual look.
4. **Reconstructed rule C01.** It reproduces the saved counts but is not the original code. A difference would affect 1 training row and 9 test houses.
5. **Registry writes are not atomic (F27).** One JSON line per `append`, no file lock. A process killed mid-write could truncate the last line; two writers could take the same id. Mitigated only by the rule that one writer owns the registry at a time.
6. **Pickled artefacts (F28).** Fitted models are joblib pickles. Load only files this project produced.
7. **Automatic decisions.** Registry decisions "retain" and "reject" come from a one-paired-standard-error rule on five folds and are provisional (ruling D14). Read the numbers, not the label.
8. **Environment.** pandas is pinned to 2.3.3 and scikit-learn to 1.9.1. Some dtype logic was written to behave the same under pandas 3, but only pandas 2.3.3 was run.
9. **Audit independence.** The auditor read the code and wrote probes, but is part of the same agent team. The external reviewer's read of the files named above is the independent check.
10. **The final model has seen the holdout.** The submitted model is refitted on all 1,460 rows after the holdout was scored once; no number in the repository measures its own error. The holdout figure applies to the development-row fit.
11. **Holdout composition.** The holdout has no house above 4,000 sq ft, so 0.106 is probably favourable compared with a sample that contains such houses; the CV estimate 0.1217 is the more conservative level.
