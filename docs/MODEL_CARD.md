# Model card: House Prices regressor E066 (CatBoost)

Format after Mitchell et al., *Model Cards for Model Reporting* (2019). Every number is traceable to `experiments/registry.jsonl`, `experiments/holdout_usage.log` or `docs/RESEARCH_REPORT.md`.

## Model details

| | |
|---|---|
| Model | Gradient-boosted decision trees, `catboost.CatBoostRegressor`, inside a scikit-learn `Pipeline` with in-fold preprocessing |
| Registry id / config hash | E066 / `be58905cb40a`; frozen in commit `95c3e86` (ruling D22) |
| Hyper-parameters | depth 6, learning rate 0.01669, 2,000 iterations, l2_leaf_reg 1.414, subsample 0.587, rsm 0.813 (Optuna TPE, 75 trials) |
| Inputs | 79 original columns after 17 deterministic cleaning rules + engineered groups `areas`, `baths`, `ages`, `quality_interactions`, `ordinal_quality`; categories one-hot encoded (unknown levels ignored) |
| Output | `log1p(SalePrice)`; converted to dollars with `expm1` |
| Training data | all 1,460 rows of Kaggle `train.csv`, no outlier removal (`outputs/models/E066_all.joblib`, not in Git) |
| Version | 1.0, November 2026 · Python 3.12, catboost 1.2.10, scikit-learn 1.9.1 (full pins in `requirements.txt`) |
| Author | Enrique Baena; implemented with Claude Code agent teams under the author's methodology and review |
| Licence | code MIT; data under the Kaggle competition terms |

## Intended use

- **Intended:** a portfolio and teaching artefact demonstrating leakage-safe validation and reproducible experimentation; producing a Kaggle submission file for the *House Prices: Advanced Regression Techniques* competition.
- **Not intended:** real-estate valuation, lending, insurance, taxation or any decision about real people or properties. The model has not been validated for those uses and gives no uncertainty estimate.

## Evaluation data and protocol

- Holdout: 219 rows (15 %), stratified by price decile, split off before any modelling and scored once after the freeze.
- Development: 1,241 rows; selection with `KFold(5, shuffle=True, random_state=42)`; finalists re-scored with `RepeatedKFold(5, 3, random_state=7)`.
- All fitted preprocessing (LotFrontage neighbourhood median, encoding) is learned inside each training fold.
- Metric: RMSE of `log1p(SalePrice)`, identical to the competition's RMSLE.

## Metrics

| Estimate | RMSLE | Notes |
|---|---|---|
| Fresh-fold CV (15 folds, not used for tuning) | **0.1217** | fold sd 0.013; **the number to quote** |
| Selection-fold CV (5 folds, used for tuning) | 0.1181 | optimistic by about 0.0036 |
| Holdout (219 rows, scored once) | 0.1065 | 95 % bootstrap interval 0.092–0.121; contains no house above 4,000 sq ft |
| Holdout absolute percentage error | median 5.3 %, mean ≈ 8 % | in dollars |
| **Kaggle test set (1,459 houses, one submission)** | **0.12271** | external confirmation; rank 1,225 of 4,717 on 2026-11-26; used for no decision |
| Baselines (selection folds) | mean 0.399 · OLS 0.160 · Lasso 0.1255 | |

An equal-weight average of Lasso and CatBoost scored 0.1207 on fresh folds (−0.0010, standard error ≈ 0.003); it was not chosen by a rule fixed before the comparison (parsimony).

## Performance by segment (out-of-fold, selection folds)

| Segment | RMSE | Bias (actual − predicted) |
|---|---|---|
| Cheapest price decile (≤ $106,500) | 0.215 | −0.097 (over-predicted) |
| Price deciles 2–9 | 0.076–0.106 | ≈ 0 |
| Most expensive decile (≥ $278,000) | 0.153 | +0.072 (under-predicted) |
| Living area ≥ 4,000 sq ft (4 houses) | 0.611 | mixed signs |
| Worst neighbourhoods: IDOTRR, Edwards, NoRidge | 0.239, 0.165, 0.162 | |

The worst 15 of 1,241 houses carry 31 % of the squared error.

## What the model relies on

Permutation importance on validation folds and SHAP on the fitted model agree on the top three: `OverallQual × TotalSF`, `OverallQual × GrLivArea`, `TotalSF`; then OverallCond, MSZoning, OverallQual, LotArea, Neighborhood, TotalBath. These are correlated: read them as "size times quality", not as a strict ranking. Figures in `outputs/figures/`.

## Limitations and risks

- **Regression toward the mean:** cheap houses over-valued, expensive ones under-valued.
- **Out-of-range houses:** very large or atypical houses (e.g. test Id 2550, 5,095 sq ft partial sale) are predicted unreliably; the training data give opposite signals for them.
- **Time and place:** one US city, sales 2006–2010; validation was random, not time-ordered. No claim about other markets, years or post-2010 prices; performance would be expected to drift.
- **Test-only anomalies:** three test rows have patterns absent from training (e.g. garage year 2207); CV cannot validate them, their predictions were checked by hand.
- **No uncertainty output:** point predictions only.
- **Fairness:** neighbourhood is a strong input. For any real use, neighbourhood-level errors (above) and potential proxy effects for protected characteristics would need a dedicated review; none was done.
- **Artefact safety:** the fitted model is a joblib pickle; load only files you produced.

## Reproduction

```powershell
python -m houseprices.evaluate --experiment E066     # selection-fold score
python -m houseprices.evaluate --experiment E076     # fresh-fold score
python -m houseprices.train --experiment E066 --rows all
python -m houseprices.predict --model outputs/models/E066_all.joblib
```
