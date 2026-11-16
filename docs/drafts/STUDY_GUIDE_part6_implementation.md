# Study guide, part 6: the Python implementation

The earlier parts quote the code where they need it. This part gives the map: which module does what, how a single experiment flows through the package, and the handful of design choices that make the results reproducible. The package is about 2,600 lines in `src/houseprices/`, with roughly 215 tests in `tests/`.

## 1. The data flow of one experiment

```
train.csv, test.csv
   │  cleaning.clean_train_test()        17 deterministic rules, no fitted statistics
   ▼
cleaned train / cleaned test
   │  validation.split_development_holdout()   219 Ids from experiments/holdout_ids.json
   ▼
development rows (1,241)        holdout rows (219, locked)
   │  features.build_features(df, groups)      row-wise arithmetic only
   ▼
X (features + Id), y_log = log1p(SalePrice)
   │  validation.cross_validate(estimator, X, y_log, cv, train_row_filter)
   │     for each fold: clone the pipeline, filter TRAINING rows, fit, predict the validation rows
   ▼
fold scores → registry.append(Experiment(...)) → experiments/registry.jsonl
```

`data.load_development_data(feature_set)` performs the first three steps and returns one object with `X`, `y_log`, `X_holdout`, `y_holdout_log`, `X_test`. It is the only entry point to the data, which is what makes it possible to audit: there is one place where the holdout is split off, and no experiment script reads the CSV files itself.

The split between "cleaning and features" (outside the fold) and "preprocessing" (inside the fold) is the central design rule. A step may run outside cross-validation only if its output for a row depends on that row alone: `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` is safe; "fill LotFrontage with the neighbourhood median" is not, because the median is computed from other rows, and if those include validation rows the validation score has seen itself. Part 1, chapter 3 develops this.

## 2. The modules

| module | role | things to know |
|---|---|---|
| `config.py` | paths, seeds (`SEED = 42`, `CV_SEED_FINALISTS = 7`), CV sizes, column groups, outlier Ids | one place for every constant |
| `cleaning.py` | the 17 rules of the original notebook as functions `rule_v01` … `rule_c01`, each returning the frame and a count; `clean_train_test()` applies them to both files | the counts are asserted in `tests/test_cleaning.py` against the original notebook's outputs |
| `features.py` | feature groups `base`, `areas`, `baths`, `ages`, `quality_interactions`, `ordinal_quality`, `anomaly_flags`; `build_features(df, groups)` | a feature set is a string such as `base+areas+baths` |
| `preprocess.py` | scikit-learn transformers that learn from the training fold: `GroupMedianImputer`, `SkewLog1pTransformer`, `KNNLotFrontageImputer`, `YearRangeClipper`; `make_preprocessor(...)` assembles a `ColumnTransformer` | each has `fit` (learn from training rows) and `transform` (apply to any rows) |
| `models.py` | builders `build_ridge`, `build_lasso`, `build_elasticnet`, `build_xgboost`, `build_lightgbm`, `build_catboost`, each returning `Pipeline([("prep", ...), ("model", ...)])`; `OPTUNA_SPACES` | a builder plus its keyword arguments fully defines a model |
| `validation.py` | `to_log_target`, `rmse_log`, `make_cv`, `cross_validate`, training-row filters, `score_holdout_once` | the scoring protocol and the holdout guard |
| `ensemble.py` | `BlendingRegressor`, `NestedStackingRegressor`, diversity metrics, `build_*_from_specs` | ensembles are ordinary estimators: their weights are learned inside `fit` |
| `registry.py` | the `Experiment` dataclass, `append`, `load_registry`, `get_experiment`, `config_hash` | append-only JSON lines |
| `evaluate.py` | `python -m houseprices.evaluate --experiment E066` | rebuilds and re-scores a registered experiment |
| `train.py`, `predict.py` | fit a registered configuration and save it; write the submission | not evidence of CV performance |

## 3. Why everything is a scikit-learn Pipeline

`cross_validate` does, for each fold, `est = clone(estimator); est.fit(X_train, y_train); est.predict(X_valid)`. Because the estimator is a `Pipeline` whose first step is the preprocessor, `fit` on the training rows fits the imputers, the skew selector, the encoder and the model in one call, and `predict` on the validation rows only applies them. There is no code path in which a transformer could be fitted on validation rows, short of writing one on purpose. Leakage safety is a property of the structure, not of discipline.

The same idea carries the ensembles. A `BlendingRegressor` is an estimator whose `fit` runs an inner 5-fold cross-validation on the rows it is given, learns the weights from those inner out-of-fold predictions, and refits the base models on all its rows. When the outer `cross_validate` calls `fit` on an outer-training fold, the weights are learned without the outer-validation rows. Nesting is automatic.

`clone` matters: it creates an unfitted copy with the same parameters, so no state survives from one fold to the next.

## 4. Builders, keyword arguments and the configuration hash

An experiment is stored as data, not as code:

```json
{"id": "E066", "builder": "houseprices.models:build_catboost",
 "builder_kwargs": {"learning_rate": 0.0166942..., "iterations": 2000, "depth": 6, ...},
 "feature_set": "base+areas+baths+ages+quality_interactions+ordinal_quality",
 "train_row_filter": "keep_all", "cv_n_splits": 5, "cv_n_repeats": 1, "seed": 42,
 "fold_scores": [0.103878, 0.132275, 0.139094, 0.102307, 0.112807], "cv_mean": 0.118072,
 "config_hash": "be58905cb40a", ...}
```

`data.resolve_callable("houseprices.models:build_catboost")` imports the function; calling it with `builder_kwargs` recreates the pipeline. `evaluate.run_registered` does exactly that, rebuilds the folds from the stored seed, and compares the new fold scores with the stored ones (tolerance 2e-4). This is the project's definition of "reproducible": a stored score counts only if this command returns `reproduces_registered_scores: true`.

`config_hash` is a hash of builder, arguments, feature set, row filter and CV specification. Two experiments with the same hash are the same experiment; the runners look the hash up before running (`_common.find_registered`) so nothing is registered twice. A finalist record differs from its source only in `cv_n_repeats` and `seed`, and so has a different hash.

Two consequences worth remembering. Keyword arguments must be plain JSON (numbers, strings, lists, dicts); that is why ensembles are described by "specs" (builder name plus arguments for each member) and not by estimator objects. And Optuna trials store the complete argument dictionary in the trial's user attributes, because some parameters are fixed outside the search space and would be lost if the record were rebuilt from `trial.params`.

## 5. The experiment runners

Scripts under `experiments/` are thin loops around two helpers in `experiments/models/_common.py`: `run_cv(builder, kwargs, feature_set, row_filter)` and `run_and_register(...)`. The longer-running ones (ablations, outlier policy, linear grids, Optuna, ensembles, interpretability) have a `--dry-run` that uses 150 rows and registers nothing.

| script | block | ids |
|---|---|---|
| `models/run_baselines.py` | mean, OLS, Ridge, default XGBoost | E001–E004 |
| `feature_ablation/run_ablation.py`, `run_lotfrontage.py` | feature groups, skew, LotFrontage imputation | E005–E048 |
| `models/run_outlier_policy.py` | outlier filters on training folds | E049–E054 |
| `models/run_linear_family.py` | 136-point linear grids | E055–E057 |
| `tuning/run_optuna_stage1.py` | Optuna Stage 1 and Stage 2 (`--stage 2`) | E058–E060, E065–E067 |
| `ensemble/run_ensembles.py` | averages, blends, nested stacks, diversity | E061–E064, E068–E075 |
| `finalists/run_finalists.py` | fresh-fold check | E076–E083 |
| `finalists/score_holdout.py` | the single holdout evaluation | (holdout log) |
| `interpretability/run_interpretability.py` | error analysis, permutation importance, SHAP | (no registry) |

## 6. Guards written as code

Several rules of the methodology are enforced by the program and tested, so that breaking them requires deliberate effort:

- `cross_validate` raises if any prediction is not finite or if a training-row filter empties a fold (audit F2, F3). A silent NaN score would otherwise be stored and sorted last.
- Training-row filters receive only the training part of a fold; a test asserts that validation indices are identical under every filter.
- `score_holdout_once` checks the Id set, refuses a second use, and logs.
- `train --rows all` raises until the holdout log is non-empty.
- `run_finalists.py` refuses seed 42.
- `predict` refuses a submission whose Ids differ from `sample_submission.csv` or whose prices are not finite and positive.

The tests in `tests/test_audit_round2.py` are the auditor's probes: each audit finding became a test that failed before the fix and passes after it.

## 7. Commands you should be able to type

```powershell
$py = "C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe"
& $py -m pytest tests                                               # about one minute
& $py -m houseprices.evaluate --experiment E066                     # reproduce the frozen model's selection-fold score
& $py -m houseprices.evaluate --experiment E076                     # reproduce its fresh-fold score (15 folds)
& $py -m houseprices.train --experiment E066 --rows all             # refit the final model
& $py -m houseprices.predict --model outputs\models\E066_all.joblib # rewrite the submission file
```

## 8. Limits of the implementation

- The registry is a text file with no locking. Two processes appending at once can take the same id; the project avoided this by allowing one writer at a time (audit F27).
- Fitted models are saved with joblib, which is pickle: load only files you created (F28), and expect them to break across library versions. The sidecar JSON records the versions.
- The Optuna database is not in Git. The per-trial CSV files are, and the registered experiments carry the winning arguments, so results are reproducible without it; the search itself is not exactly re-creatable because the sampler was re-seeded when Stage 2 resumed (F31).
- Thread count affects the speed of the boosters, and in principle the last digits of their results. Registered scores reproduced to four decimals on the same machine; a different machine may differ slightly.

## Self-check questions

1. Why may `TotalSF` be computed before cross-validation while the LotFrontage median may not?
2. What exactly does `python -m houseprices.evaluate --experiment E066` prove, and what does `train` not prove?
3. Why are ensemble members passed as "specs" instead of estimator objects?
4. How does the structure of `Pipeline` plus `clone` prevent a scaler from seeing validation rows?

## Answers

1. `TotalSF` for a house uses only that house's columns. A neighbourhood median uses other houses; computed on all rows it would include validation houses in a statistic used to fill training and validation rows alike.
2. That the stored configuration, rebuilt from the registry and scored on folds rebuilt from the stored seed, gives the stored fold scores. `train` only produces a fitted object; it says nothing about how well the procedure generalises.
3. Registry arguments must be JSON so that they can be hashed, stored and rebuilt; a spec (builder name plus JSON arguments) can be, a Python object cannot.
4. `cross_validate` clones the whole pipeline for each fold and calls `fit` with training rows only; the scaler is a step of that pipeline, so its statistics come from those rows. On validation rows only `transform` is called.
