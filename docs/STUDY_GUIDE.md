# Study guide: Kaggle House Prices, from a cleaning notebook to a frozen, audited model

This guide explains everything the project did, why it did it, and how the code does it, in enough detail for you to defend each decision without the repository open. It is long on purpose. Read it in order once; afterwards the self-check questions at the end of each part are the fastest way to revise.

## How the guide is organised

| Part | Topic | Written by |
|---|---|---|
| 1 | The data, the 17 cleaning rules, fitted statistics and leakage, the engineered feature groups, outliers, test-only anomalies | feature-researcher |
| 2a | Validation methodology: the metric, K-fold CV, selection bias, a leakage taxonomy, reproducibility, what the audits found | feature-researcher |
| 2b | Linear models: OLS, Ridge, Lasso, ElasticNet, skew correction, scalers, the grid results | feature-researcher |
| 2c | Boosted trees and Optuna: trees, gradient boosting, XGBoost, LightGBM, CatBoost, TPE, the staged tuning budget, Stage 1 and Stage 2 | model-researcher |
| 3 | Ensembles: why averaging helps, learned blend weights, nested stacking, diversity, the final ensemble results, finalists on fresh folds | ensemble-researcher |
| 4 | From finalists to a frozen model: the decision rule, the single holdout evaluation, the final fit and the submission file | director |
| 5 | What the final model learned and where it fails (interpretability and error analysis) | model-researcher |
| 6 | The Python implementation: how the package fits together and how to run everything | director |
| 7 | Defending the project: the decision list, what was rejected, the limitations, and a one-page mental model | director |

The parts were written by different agents at different moments of the campaign and were merged with light editing. Where an early part describes a state that later changed (for example "Stage 1" results in part 3), the text says so and points to the section with the final numbers. The authoritative record of every number is the experiment registry (`experiments/registry.jsonl`) and the decision log (`docs/EXPERIMENT_HISTORY.md`, decisions D1 to D22).

## The whole project in one page

**Problem.** Predict the sale price of 1,459 houses in Ames, Iowa, from 79 descriptive columns, learning from 1,460 houses with known prices. Kaggle scores the root mean squared error of the logarithm of the price (RMSLE). The project predicts `log1p(SalePrice)` and reports RMSE on that scale, which is the same number.

**Protocol.** 219 training houses (15 %, stratified by price decile) were set aside once as a holdout and locked in code. Every experiment used the other 1,241 "development" rows with a shuffled 5-fold cross-validation (seed 42). Everything that learns from data (imputation statistics, the choice of skewed columns, encoders, scalers, models, ensemble weights) is fitted inside each training fold. Candidates for the final model were re-scored on 15 new folds (3 repeats of 5 folds, seed 7) because scores on the folds used for choosing are optimistic. Then one procedure was frozen, approved by an independent auditor, and scored on the holdout exactly once.

**The ladder of results** (RMSE on the log scale; lower is better):

| step | what changed | selection folds | id |
|---|---|---|---|
| predict the mean for everybody | nothing learned | 0.3987 | E001 |
| ordinary least squares on numeric columns | first real model | 0.1599 | E002 |
| Ridge on one-hot categories | categories + shrinkage | 0.1409 | E003 |
| Ridge + engineered features + in-fold skew correction | feature engineering | 0.1304 | E041 |
| Lasso, tuned alpha | sparsity | 0.1255 | E056 |
| CatBoost, 25 Optuna trials | boosted trees | 0.1195 | E060 |
| CatBoost, 75 Optuna trials | more tuning | 0.1181 | E066 |
| average of Lasso and CatBoost | ensemble | 0.1173 | E068 |

**The honest numbers.** On the fresh folds the frozen model, single CatBoost E066, scores 0.1217 and the Lasso + CatBoost average 0.1207; the difference (0.0010) is a third of its standard error. On the single-use holdout the frozen model scores 0.1065. The project's claim is: expected RMSLE about 0.12, with the holdout as a consistency check and not as a better estimate.

**Three results that go against common practice in public notebooks.**
1. Removing the famous outliers made every model worse once the removal was restricted to training folds (part 1, chapter 5).
2. The 75-trial tuned CatBoost was not significantly improved by any ensemble; learned blend weights and stacking never beat a plain 50/50 average (part 3).
3. Every score chosen on the selection folds dropped by 0.003 to 0.004 when measured on new folds; an analysis that reports the selection score is reporting a number that is too good (parts 2a and 4).


---

## Study guide, part 1: the data, the cleaning and the features

Author: feature-researcher. Audience: a third-year data-science student who must be able to defend every decision in an interview. Numbers come from `docs/AUDIT_2026-11-11.md`, `docs/EXPERIMENT_HISTORY.md` and the experiment registry (experiment ids are written E0xx). Cross-validated errors are RMSE on log1p(SalePrice), five shuffled folds, seed 42, lower is better.

---

### Chapter 1. The Ames data and the competition metric

**The data.** The Ames Housing data describe 2,919 house sales in Ames, Iowa between 2006 and 2010. Kaggle splits them into `train.csv` (1,460 houses with the sale price) and `test.csv` (1,459 houses without it). There are 79 recorded attributes: lot size and shape, the neighbourhood, build and remodel years, quality and condition ratings from 1 to 10, areas of floors, basement, garage and porches, counts of rooms and bathrooms, and many categorical descriptors. The target is `SalePrice`.

**Why the metric is computed on logarithms.** The competition scores the root mean squared error between the logarithm of the predicted price and the logarithm of the actual price (often called RMSLE). Three reasons make this the right choice and not just a rule of the contest.

1. **Errors are relative.** A difference of log-prices is a ratio of prices. Being wrong by $20,000 on a $100,000 house is a serious miss, and the same $20,000 on a $500,000 house is minor. A log-scale error treats both by their percentage, which is how buyers and appraisers think.
2. **The target is right-skewed.** Raw prices have skewness 1.88, with a long tail of expensive houses. After `log1p` the skewness is 0.12, close to symmetric. Squared-error models behave much better on a symmetric target, and a handful of mansions stop dominating the loss.
3. **Prices are multiplicative.** A garage adds a percentage to a price, not a fixed amount. In log space multiplicative effects become additive, which is exactly what a linear model assumes.

We use `log1p(x) = log(1 + x)` rather than `log(x)`. For prices the two are almost identical, but `log1p` is the exact inverse partner of `expm1`, never fails at zero and is numerically careful for small values. Predictions are converted back with `expm1`, which also guarantees every predicted price is positive.

**What a 0.01 difference means in dollars.** An RMSE of 0.01 in log space means a typical error of about 1% of the price. At the median training price of $163,000 that is roughly $1,600. The table converts several levels of error to dollars at that median.

| RMSE (log1p) | Typical relative error | Dollars at the $163,000 median | Where it appears in our results |
|---|---|---|---|
| 0.399 | about 49% | about $80,000 | predicting the mean for everyone (E001) |
| 0.141 | about 15% | about $24,500 | Ridge on base features (E003) |
| 0.126 | about 13% | about $21,800 | best linear result of the feature stage, ElasticNet on all features with skew correction (E042); the final CatBoost model is at about 0.12 (parts 2c and 4) |
| 0.010 | about 1% | about $1,600 | the size of a difference we would call tiny |

Two cautions you should be able to state. First, the dollar translation is approximate because the error is a ratio: the same log error is a bigger dollar error on an expensive house. Second, a model trained in log space and converted back with `expm1` predicts something close to the median price of houses like this one, not the mean. For a pure ranking metric like RMSLE that is fine, but it is a systematic bias if someone reads the predictions as expected dollar amounts.

**The reference level.** The mean baseline scores 0.3987, which is simply the standard deviation of log-price. Any model must be judged against that number: a score near 0.4 means the model learned nothing.

---

### Chapter 2. The seventeen cleaning rules

#### 2.1 The principle: semantic fills, not statistical fills

In this data a missing value usually does not mean "unknown measurement". It often means the feature does not exist. `PoolQC` is blank for 1,453 of 1,460 houses because those houses have no pool, and `data_description.txt` says so. Replacing that blank with the most common quality grade would invent pools. The cleaning rules therefore fill by **meaning**: "NoPool", "NoBasement", "NoGarage", and so on. A semantic fill has two properties we rely on.

- It uses only the same row (and the data dictionary). Nothing is estimated from other houses, and `SalePrice` is never read. So the rules cannot leak information between training and validation data and may safely run once, before any cross-validation split.
- It never overwrites a recorded value. Rules only fill a missing entry or add a flag column. Contradictions (for example a pool area above zero but no pool quality) are flagged and marked "Unknown", not "fixed".

A **statistical** fill, such as replacing a missing `LotFrontage` with a median, is different: it is estimated from data and depends on which rows you look at. Those fills are deliberately left out of the cleaning step and done later inside each training fold (Chapter 3).

A technical detail called rule **P01** makes several rules possible. pandas treats the text "NA" and "None" as missing when it reads a CSV. But in this dataset "None" is a valid veneer category and "NA" is a marker meaning "feature absent". The cleaning code therefore reads each file twice, once normally and once with `keep_default_na=False`, and uses the raw copy to tell the cases apart.

#### 2.2 The rules by family

Counts are the number of houses (or entries where stated) a rule changed or flagged, as verified in the original notebook output and asserted in `tests/test_cleaning.py`.

**Masonry veneer**

| Rule | Trigger | Action | Train | Test | Why semantic |
|---|---|---|---|---|---|
| V01 (with P01) | `MasVnrType` reads as missing but the raw text is "None" | set to "None", a valid category | 864 | 878 | The raw file says "None"; pandas misread it. |
| V02 | missing and raw text is "NA" | set to "Unknown"; area untouched | 8 | 16 | The data does not say; we do not guess a material. |
| V03 | type and area disagree ("None" with area above 0, or a material with area 0) | add flag `MasVnrConflict`; change no value | 7 | 3 | Two recorded values contradict each other; we mark it, not choose. |

**Basement**

| Rule | Trigger | Action | Train | Test | Why semantic |
|---|---|---|---|---|---|
| B01 | `TotalBsmtSF` is 0 and all five basement categoricals are missing | set them to "NoBasement" | 37 | 41 | An area of zero confirms there is no basement. |
| B02 | basement area above 0 but some descriptor missing | fill only the missing ones with "Unknown" | 2 | 7 | A basement exists but a descriptor was not recorded. |
| B03 | house is "NoBasement" and a bath count is missing | count becomes 0 | 0 | 1 | No basement means no basement bathrooms. |
| B04 | every basement field missing (area and counts included) | categoricals "Unknown", flag `BasementStatusUnknown` | 0 | 1 | We do not even know whether a basement exists. |

**Garage**

| Rule | Trigger | Action | Train | Test | Why semantic |
|---|---|---|---|---|---|
| G01 | `GarageCars` 0, `GarageArea` 0 and four garage categoricals missing | set to "NoGarage"; `GarageYrBlt` stays numeric and missing | 81 | 76 | Zero capacity and area confirm no garage. |
| G02 | derived indicators | add `HasGarage` and `GarageYearUnknown` | none | 2 | A garage exists but its year is absent. |
| G03 | garage exists but finish, quality or condition missing | fill only the missing with "Unknown" | 0 | 2 | Existence is known, the descriptor is not. |

**Absence of a feature (pool, fireplace, extras)**

| Rule | Trigger | Action | Train | Test |
|---|---|---|---|---|
| A01 | `PoolQC` missing and `PoolArea` 0 | "NoPool" | 1,453 | 1,453 |
| A02 | `FireplaceQu` missing and `Fireplaces` 0 | "NoFireplace" | 690 | 730 |
| A03 | raw `MiscFeature` is "NA" and `MiscVal` is 0 | "NoMiscFeature" | 1,406 | 1,407 |
| A04 | raw `Alley` is "NA" | "NoAlleyAccess" | 1,369 | 1,352 |
| A05 | raw `Fence` is "NA" | "NoFence" | 1,179 | 1,169 |
| A06 | contradictions: pool area above 0 without a quality, or `MiscVal` above 0 without a feature | "Unknown" plus flag `MiscFeatureConflict` | 0 | 3 pools, 1 conflict |

All of these are semantic for the same reason: another recorded column (an area, a count, a raw "NA") proves the feature is absent, so the label is a fact, not an estimate. A06 is the mirror case, where the other column proves the feature is present but its description is missing.

**Remaining categorical gaps**

| Rule | Trigger | Action | Train | Test |
|---|---|---|---|---|
| C01 | any categorical gap left (`Electrical`, plus seven columns that are missing only in test) | "Unknown" | 1 entry in 1 house | 12 entries in 9 houses |

C01 is a catch-all. "Unknown" is itself a category. A model sees it as one more level, and for a column that is almost never missing it simply carries little weight.

#### 2.3 The C01 story: a reconstruction, not a recovery

In the original notebook the cell that defined the list of columns used by C01 (`c01_cols`) was lost. Cell 65 still referenced it, the execution counters jumped from 59 to 63, and a clean run raised a `NameError`. The code is missing in every available copy, including an older backup. What survives is (a) the written description in the cleaning log and (b) the saved verification output, which says C01 touched 1 entry in 1 house in train and 12 entries in 9 houses in test.

We therefore wrote the rule again from that description and **labelled it as a reconstruction** everywhere (docstring, tests, notebook). The test suite checks that the reconstruction reproduces exactly the documented counts. A reconstruction that matches the saved evidence is good enough to continue, but you must never say "we recovered the code". In an interview this is a good example of honest provenance: say what you know, what you rebuilt and how you verified it.

#### 2.4 B04: a condition instead of an Id

The original notebook hard-coded the basement-unknown case as `Id == 2121`. That works for this one file but encodes a fact about a row, not a rule. We replaced it with a condition: all basement fields are missing. The condition selects exactly Id 2121 in test and nothing in train, and a test asserts both. The benefit is that the same rule will behave correctly on any new data, and it is honest about why the house is special.

A related side-effect: the flags `BasementStatusUnknown`, `GarageYearUnknown` and `MiscFeatureConflict` are constant (all False) in training. They are removed from the feature set (`config.CONSTANT_IN_TRAIN_FLAGS`), because a column that never varies cannot teach a model anything. They still document the test rows that need care.

---

### Chapter 3. What a "fitted statistic" is and why it lives inside the fold

#### 3.1 The definition

A **fitted statistic** is any number a preprocessing step computes from data and then reuses: a median, a mean, a category list, a scaling range, "which columns are skewed". The test is simple: if the number would change when you show the step a different set of rows, it is a fitted statistic.

Compare two operations.

- "If `PoolArea` is 0 and `PoolQC` is missing, write NoPool." Whatever other rows you show, the answer for this row is the same. Not fitted. May run once, anywhere.
- "Replace a missing `LotFrontage` with the median frontage of this house's neighbourhood." The median depends on which houses you include. Fitted.

#### 3.2 Why fitted statistics must live inside the fold

In cross-validation the validation fold plays the role of future, unseen houses. If a median or a skewness is computed using the validation rows, the training procedure has indirectly seen them. The score becomes optimistic: that is **leakage**. The effect for imputation medians is small, but the habit matters, and for target-dependent steps it can be enormous. The discipline is: **every fitted step is refit on each training fold and only applied to the validation fold.** In scikit-learn this is achieved by putting the preprocessing and the model in one `Pipeline`; cross-validation then clones and refits the whole pipeline per fold.

#### 3.3 The three fitted components in this project

| Component | What it learns from the training fold | What is applied to unseen rows |
|---|---|---|
| `GroupMedianImputer` | the median `LotFrontage` of each neighbourhood, plus a global median as fallback | fills a missing frontage with its neighbourhood median, or the global median if the neighbourhood is new or all-missing |
| `SkewLog1pTransformer` | which numeric columns have skewness above 0.75 and minimum at least 0 | applies `log1p` to exactly those columns, whatever the unseen rows look like |
| `KNNLotFrontageImputer` | a standardiser plus the training houses (their lot area, floor area, quality, year built, frontage) | fills a missing frontage from the five nearest training houses |

The skew transformer is the subtle one. The choice of columns is a statistic, so it is learned from the training fold. Two folds can therefore pick different columns. The unit tests check exactly that: the selected columns differ between two different samples, and `transform` on unseen rows uses the fitted list and does not re-measure skewness.

The scaler and the one-hot encoder inside `make_preprocessor` are fitted statistics too (medians, interquartile ranges, category lists). Unseen categories at prediction time are ignored by the encoder rather than raising an error.

#### 3.4 Why the cleaning rules may run once

The seventeen rules are not fitted. Each decision for a house uses only that house's own columns and the fixed data dictionary. We run them on the full training and test tables once, identically for both. The tests also confirm the rules are idempotent (running them twice changes nothing) and that they never touch `SalePrice`. This is why `clean_train_test` is called once in `data.load_development_data`, while the preprocessor is rebuilt inside every fold.

One more rule of the same family: the **feature groups** of Chapter 4 are pure row-wise formulas, so they may also be built once. Their missing results (for example `GarageAge` for houses with no garage) are left as missing and filled later, inside the fold.

---

### Chapter 4. The engineered feature groups

#### 4.1 How the ablation was run

We tested each group by adding it alone to `base` ("base+group"), by adding all five ("all"), and by removing each group in turn from "all" (leave-one-out). Every configuration was scored with Ridge (alpha 10), ElasticNet (alpha 0.001, l1 ratio 0.5) and XGBoost with default settings, using the same five folds. We compare configurations with a **paired** difference: for each fold we subtract the two scores and look at the mean and the standard error (SE) of those five differences. Pairing matters because the folds differ a lot in difficulty (Ridge fold scores range from 0.112 to 0.185 on base). A difference smaller than about one paired SE, and certainly below 0.001, is noise.

Reference scores on base: Ridge 0.14088 (E003), ElasticNet 0.13786 (E005), XGBoost 0.14868 (E004).

#### 4.2 Group by group

| Group | Columns and formula | Motivation |
|---|---|---|
| areas | `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`; `TotalPorchSF` = sum of the five porch and deck areas | Price grows with total space; one total is easier than three parts. |
| baths | `TotalBath = FullBath + 0.5 HalfBath + BsmtFullBath + 0.5 BsmtHalfBath` | Four sparse counts become the number buyers care about. |
| ages | `HouseAge = max(YrSold - YearBuilt, 0)`, `RemodAge = max(YrSold - YearRemodAdd, 0)`, `GarageAge = YrSold - GarageYrBlt` only when the garage year is present and not after the sale, else missing | Depreciation depends on age at sale; a linear model cannot subtract two columns itself. |
| quality_interactions | `OverallQual x GrLivArea`, `OverallQual x TotalSF` | A quality premium scales with house size. |
| ordinal_quality | `ExterQual_ord`, ... (ten columns): Po 1, Fa 2, TA 3, Gd 4, Ex 5, absence 0, "Unknown" left missing | One monotone slope instead of five unrelated one-hot columns. |

**What each group did** (change in cv_mean versus the same model on base; negative is better).

| Group | Ridge | ElasticNet | XGBoost | Experiment ids |
|---|---|---|---|---|
| areas | +0.0001 | +0.0001 | -0.0034 | E006, E007, E008 |
| baths | 0.0000 | -0.0003 | -0.0055 | E009, E010, E011 |
| ages | +0.0001 | 0.0000 | -0.0021 | E012, E013, E014 |
| quality_interactions | -0.0023 | -0.0029 | -0.0099 | E015, E016, E017 |
| ordinal_quality | +0.0003 | -0.0001 | -0.0017 | E018, E019, E020 |
| all five | -0.0022 | -0.0028 | -0.0131 | E021, E022, E023 |

**Leave-one-out from "all"** (change versus "all"; positive means removing the group hurts, so the group helps).

| Removed group | Ridge | ElasticNet | XGBoost | Ids |
|---|---|---|---|---|
| areas | +0.0003 | +0.0004 | +0.0012 | E024 to E026 |
| baths | 0.0000 | +0.0002 | +0.0020 | E027 to E029 |
| ages | 0.0000 | 0.0000 | -0.0021 | E030 to E032 |
| quality_interactions | +0.0026 | +0.0025 | +0.0086 | E033 to E035 |
| ordinal_quality | -0.0004 | -0.0006 | +0.0012 | E036 to E038 |

#### 4.3 The lesson: quality_interactions carried the signal, ordinal_quality did not

**quality_interactions** is the only group whose removal clearly hurts every model, by 0.0025 to 0.0086, and it also improves every model when added alone. The reason is mechanical. A linear model adds effects: it can say "each quality point is worth x" and "each square foot is worth y" but not "a quality point is worth more in a bigger house". The product column supplies exactly that. XGBoost can in principle build such interactions from splits, but it needs many splits to approximate a smooth product, so a ready-made column helps it most (0.0099).

**ordinal_quality** looked like it should help, because quality labels are ordered, and in the one-hot form the model treats Ex and Gd as unrelated. But those same ten columns are already represented, and the one-hot columns of the same labels stay in the data (the group keeps the text columns). The numeric twin therefore adds a second, partly redundant encoding. Alone it moves Ridge and ElasticNet by +0.0003 and -0.0001, which is noise. Removing it from "all" slightly improves the two linear models (0.0004 and 0.0006) but slightly hurts XGBoost (0.0012). Neither direction exceeds one paired SE for Ridge and XGBoost. The lesson is general: **a feature that re-encodes information the model already has adds nothing, while a feature that gives the model a relationship it cannot build (a product) adds a lot.**

**The others.** areas, baths and ages are individually small for linear models (0.0003 or less), but give XGBoost something: baths helps it by 0.0055. Totals save the trees several splits. At default hyper-parameters this evidence is weak; tuning may change it, which is why the registry decisions are marked provisional (D14).

#### 4.4 The ruling D11 and why we did not prune

The best linear results at default alpha came from removing ordinal_quality, and the best XGBoost result from removing ages. We did not prune per model. Every leave-one-out variant lies within about 0.001 of "all" except the removal of quality_interactions. Choosing the winner per model from differences that size, with untuned hyper-parameters and five folds, would be fitting to fold noise. The working set is the full "all" (D11), and the ordinal_quality question is re-examined on the finalists with RepeatedKFold 5x3.

#### 4.5 The skew transform: the biggest single gain

The in-fold `log1p` of skewed non-negative numeric columns improved the linear models by about 0.009 and was consistent across feature sets (D12).

| Configuration | Ridge | ElasticNet | Ids |
|---|---|---|---|
| base | 0.14088 | 0.13786 | E003, E005 |
| base with skew | 0.13192 | 0.12845 | E039, E040 |
| all | 0.13871 | 0.13503 | E021, E022 |
| all with skew | 0.13042 | 0.12562 | E041, E042 |

The gain is larger than any feature group (0.0083 to 0.0094 versus at most 0.0029 for the best group). The mechanism: a column like `LotArea` has skewness 13 because a few giant lots exist; after `log1p` it is about -0.3. A linear fit is then no longer controlled by those few lots. Trees only use the order of values, and `log1p` does not change order, so XGBoost does not receive the transform.

#### 4.6 LotFrontage: a non-issue

`LotFrontage` is missing for about 17% of houses. Neighbourhood median (E043, E046), global median (E044, E047) and KNN (E045, E048) all land within noise.

| Method | Ridge | XGBoost |
|---|---|---|
| neighbourhood median | 0.14088 | 0.14868 |
| global median | 0.14099 | 0.14701 |
| KNN | 0.14048 | 0.14900 |

Ridge differences are at most 0.0005. XGBoost's global median is 0.0017 better than the neighbourhood median, within one paired SE. We keep the neighbourhood median (D13), the documented choice of the original cleaning log, because nothing more elaborate beats it. The general lesson: when a column carries little signal after other columns are present (here lot area already proxies frontage), a cleverer fill rarely moves the score.

---

### Chapter 5. The outlier-policy lesson (D15)

**Background.** Two training houses, Ids 524 and 1299, are very large (`GrLivArea` above 4,000 square feet) but sold cheaply, in Edwards, as partial sales. Almost every public notebook for this competition deletes them. Our EDA noticed them but did not delete them. Ids 692 and 1183 are also large but legitimately expensive.

**The honest test.** We evaluated three policies under cross-validation where **only the training part of each fold is filtered** (D3): keep everything; drop Ids 524 and 1299; drop every house above 4,000 square feet (which also drops 692 and 1183). Validation folds are always complete.

| Policy | Ridge | ElasticNet | XGBoost |
|---|---|---|---|
| keep_all | 0.13042 | 0.12562 | 0.13561 |
| drop Ids 524 and 1299 | 0.13480 (+0.0044) | 0.13205 (+0.0064) | 0.14062 (+0.0050) |
| drop GrLivArea above 4000 | 0.13603 (+0.0056) | 0.13417 (+0.0085) | 0.13641 (+0.0008) |

Keep-all is best for all three models. Dropping the two EDA outliers is worse beyond one paired SE for ElasticNet and XGBoost. (Experiments E049 to E054.)

**Why public notebooks that drop these Ids look better than they deserve.** If you delete the two houses from the whole table before splitting, they disappear from the validation folds as well. Those houses are the hardest to predict (a very large house that sold cheaply), so their absence from validation lowers the error. The model did not become better; the exam became easier. In our honest protocol, Ids 1299, 692 and 1183 fall in validation fold 1 and Id 524 in fold 2. A model trained without them is still graded on them, and the fold standard deviation of the linear models rises from about 0.023 to about 0.035.

There is a second point that cuts the other way and that you should know. The test set contains one house of exactly this profile: Id 2550 (5,095 square feet, Edwards, partial sale, quality 10, and also one of the three anomalies of Chapter 6). Whether a model trained with or without Ids 524 and 1299 predicts it better cannot be settled by cross-validation, which contains no such comparison on unseen data of this kind. We therefore state the finding narrowly: under honest validation, dropping the two houses did not improve any model; the prediction for Id 2550 should be inspected by hand (Chapter 6).

**What to say in an interview.** "Outlier removal is a modelling choice and must be evaluated like one: filter training rows inside the fold, keep validation rows whole. Done that way, none of our policies beat keeping everything, so the final model keeps all rows (D15)." Do not say outlier removal is wrong in general. Say it was not supported by this evidence, with these models.

---

### Chapter 6. The anomaly flags, and test-only anomalies at prediction time

#### 6.1 The finding

Three audit flags mark impossible years: `GarageYrBltImplausible` (garage built after the sale), `SoldBeforeBuilt` and `RemodBeforeBuilt`. They are produced by `cleaning.add_anomaly_flags`, which only adds columns.

| Flag | Train | Test | Test Id |
|---|---|---|---|
| GarageYrBltImplausible | 0 | 2 | 2593 (garage year 2207), 2550 |
| SoldBeforeBuilt | 0 | 1 | 2550 (sold 2007, built 2008) |
| RemodBeforeBuilt | 0 | 1 | 1877 (remodelled 2001, built 2002) |

All three flags are zero for every training house. They are non-zero only for three test houses. A column that is constant in the training data cannot be learned from, so `base+anomaly_flags` can never differ from base. We kept the group in `FEATURE_SETS` for documentation but excluded it from the default ablation to save budget. This is a real data finding: **the training data contain no examples of this kind of anomaly, so no model trained on it can know what to do with them.**

#### 6.2 What to do at prediction time

The options, in order of how defensible they are:

1. **Never delete the rows and never edit recorded values.** Kaggle needs a prediction for every test Id, and rewriting 2207 to something else would be an invented fact.
2. **Make features robust, row by row.** The `ages` group already does this: a house sold before it was built gets `HouseAge` 0 (not negative), and an impossible garage year gives a missing `GarageAge`, which the in-fold imputer fills with a training median. So the derived features cannot produce nonsense for the three houses.
3. **Guard the one remaining raw impossible value with a fitted, opt-in step.** The raw column `GarageYrBlt` is part of the base features and equals 2207 for Id 2593. A linear model multiplies it by a coefficient learned on garage years between 1900 and 2010, so that single value would be an extrapolation of about 200 years and could shift the prediction. We implemented `YearRangeClipper` in `preprocess.py` and switch it on with `make_preprocessor(clip_years=True)`. In `fit` it learns, for each year-like numeric column present (`YearBuilt`, `YearRemodAdd`, `GarageYrBlt`; `YrSold` is routed to the categorical branch and never reaches it), the **maximum** seen in the training fold. In `transform` it caps larger years at that maximum and leaves missing values missing. It runs first in the numeric branch, before skew correction, imputation and scaling.
   - **Upper bound only (ruling D20).** A year beyond the training maximum is impossible for this data, so it is a data-entry error. An old year is not: Ids 1603 and 2218 have garages built in 1896 and 1895, which are real values that merely fall below the training minimum of 1900. A two-sided clip (`lower=True`, off by default) would move those two by 4 and 5 years and distort true information, so the default does not touch the lower side.
   - **It is provably a no-op on development rows.** Every year in the development data lies inside the training range, so within cross-validation the step changes nothing. A unit test fits a Lasso (the E056 configuration) with and without the clip and shows identical predictions, so no CV score changes and registered experiments still reproduce.
   - **On the test frame it changes exactly one value:** Id 2593, `GarageYrBlt` 2207 becomes 2010. A test asserts that this is the only difference.
   - **Why this is a fitted step and not a cleaning rule.** The cap is the training maximum, a statistic of the data, so it belongs inside the preprocessor and is refit per fold. It is also why the original CSV is never edited: the recorded 2207 stays visible in the cleaned table and the flag `GarageYrBltImplausible`, and only the model input is capped.
   - **What remains uncertain.** Id 2550 (5,095 square feet, quality 10, sold before it was built) is not repaired by the clipper, and no preprocessing can say what such a house is worth: it is a giant house of a kind the training data barely contain (like Ids 524 and 1299). This is an irreducible uncertainty. Check its predicted price by hand in the submission file against similar large houses and note it as a known risk.
4. **Log it.** The final prediction script should print the three Ids and their predicted prices so the choice is auditable.

---

### Self-check questions

1. Why does the competition use the logarithm of price, and why `log1p` rather than `log`?
2. A model has RMSE 0.126 on log1p(price). Roughly what does that mean for a house at the median price, and why is the dollar figure only approximate?
3. Explain the difference between a semantic fill and a statistical fill. Give one example of each from this project.
4. Why must the cleaning code read the CSV twice, and what does rule P01 protect?
5. Why can the cleaning rules run once on the whole data while the LotFrontage median must be refit in every fold?
6. The code for rule C01 was lost. What did we do, and what must we never claim?
7. `SkewLog1pTransformer` selects the columns to transform. Why is this selection a leakage risk if done on the whole table, and what does the test suite check?
8. Which feature group mattered most, why could a linear model not build it, and which group did nothing?
9. Why do public notebooks that drop Ids 524 and 1299 report better cross-validation scores than they deserve, and what did our honest protocol show?
10. The anomaly flags are all zero in training. What does that imply for a model that uses them, and what would you do for test Id 2593 at prediction time?
11. For the impossible garage year 2207 (test Id 2593), why do we clip inside the fold with a fitted step instead of simply editing the CSV, and why only the upper bound?

### Answers

1. The error is a ratio: a $20,000 miss matters on a cheap house and not on an expensive one. The raw target has skewness 1.88 and the log reduces it to 0.12, and prices combine multiplicatively, which becomes additive in log space. `log1p` is the exact inverse of `expm1`, is safe at zero and keeps predictions positive after conversion back.
2. About 13% typical relative error, which is roughly $21,800 at the $163,000 median. It is approximate because the same log error means a larger dollar error on an expensive house, and because converting a log prediction back gives roughly the median price, not the mean.
3. A semantic fill uses meaning from the same row and the data dictionary, for example "NoPool" when `PoolArea` is 0. A statistical fill estimates a value from other rows, for example the neighbourhood median of `LotFrontage`. Semantic fills are leak-free and run once; statistical fills must be fitted inside each training fold.
4. pandas turns the text "NA" and "None" into missing values, but "None" is a valid veneer category and "NA" means the feature is absent. P01 keeps a raw copy read with `keep_default_na=False` so rules can tell a valid category from a true gap.
5. Each cleaning decision depends only on that house and the fixed data dictionary, so the answer cannot change with the rows shown. A median depends on which rows you include, so computing it with validation rows leaks information. It is refit per fold inside the pipeline.
6. We re-implemented the rule from the written cleaning log and the saved verification output, labelled it a reconstruction and tested that it reproduces 1 entry in 1 house (train) and 12 entries in 9 houses (test). We must never say we recovered the original code.
7. Which columns are skewed is a statistic of the data. Computing it on the full table lets validation rows influence the choice. The tests check that the selected columns differ between two different training samples and that `transform` applies the fitted list to unseen rows without re-measuring skewness.
8. quality_interactions mattered most: removing it from "all" hurts every model by 0.0025 to 0.0086. A linear model adds effects and cannot form "quality times size" by itself. ordinal_quality did essentially nothing because it re-encodes information the one-hot columns already carry.
9. Deleting the two houses before splitting removes them from the validation folds too. They are the hardest cases, so the exam becomes easier and the error falls without the model improving. When only the training part of each fold is filtered, keeping all rows was best for all three models (D15).
10. A constant column has no variation to learn from, so the model gets no information from it, and the ablation run could not differ from base. At prediction time we keep the row, do not edit the 2207, rely on the `ages` group to give a missing `GarageAge` that is imputed in-fold, check that the raw `GarageYrBlt` value does not distort a linear prediction, and log the prediction for review.
11. Editing the CSV would overwrite a recorded value with an invented one and hide the anomaly. The cap is the largest year seen in the training fold, which is a fitted statistic, so it belongs in the preprocessor where it is refit per fold; on the development rows it is a no-op, so CV is untouched. Only the upper bound is applied because years beyond the training maximum are impossible, while an old year such as the 1895 and 1896 garages of Ids 1603 and 2218 is a real value that clipping would distort.


---

## Study guide, part 2a: validation methodology

Author: feature-researcher. Same audience as part 1: a third-year data-science student who must defend every decision in an interview. Function and file names are quoted from `src/houseprices/validation.py`, `registry.py`, `evaluate.py`, `config.py` and from `docs/VALIDATION_AUDIT.md` and `docs/EXPERIMENT_HISTORY.md` as read on writing. Decision numbers (D3, D14, D15, D17) and finding numbers (F1 to F28) refer to those documents.

---

### 1. The metric

#### 1.1 Definition and equivalence with RMSLE

For houses i = 1..n with actual price a_i and predicted price p_i, the competition metric is

RMSLE = sqrt( (1/n) * sum_i ( log(1 + p_i) - log(1 + a_i) )^2 ).

We train on y_i = log(1 + a_i) (the function `validation.to_log_target`, which is `numpy.log1p`) and a model predicts yhat_i on that same scale. Then

RMSE on the log scale = sqrt( (1/n) * sum_i ( yhat_i - y_i )^2 ),

and when yhat_i = log(1 + p_i) the two expressions are **identical**. That is the whole equivalence: the competition metric is the ordinary RMSE of a model fitted on log1p(price). The code function is `validation.rmse_log`; `validation.rmsle_dollars` computes the metric from dollar values as a cross-check, and `validation.from_log_target` (`expm1`) converts predictions back to dollars. Using the log scale for training and scoring at once means the model optimises exactly the quantity it is judged on.

#### 1.2 Why it penalises relative errors

The difference of two logarithms is the logarithm of a ratio:

log(1 + p) - log(1 + a) = log( (1 + p) / (1 + a) ).

For prices in the tens of thousands of dollars the "+1" is negligible, so this is log(p / a). A prediction that is 10% too high has error log(1.10) = 0.095 whatever the price level; a prediction 10% too low has log(0.90) = -0.105. So a $20,000 miss on a $100,000 house (20%) costs about log(1.2) = 0.18, while the same $20,000 on a $400,000 house (5%) costs about 0.049. The metric counts percentages, not dollars. Two consequences: cheap houses carry as much weight as expensive ones, and an under-prediction by a fixed percentage costs slightly more than an over-prediction by the same percentage (0.105 versus 0.095 at 10%), because the logarithm is asymmetric.

#### 1.3 What an RMSE of 0.12 means in dollars

An RMSE of r on the log scale means a typical multiplicative error of e^r. For r = 0.12: e^0.12 = 1.1275 on the high side and e^-0.12 = 0.8869 on the low side, that is about +12.7% or -11.3%.

| Price level | High-side error (+12.7%) | Low-side error (-11.3%) |
|---|---|---|
| Median price, $163,000 | about +$20,800 | about -$18,400 |
| $400,000 | about +$51,000 | about -$45,200 |

Our current best linear-plus-skew result is 0.1256 (E042) and the mean baseline is 0.3987, the standard deviation of log-price. State it as "a typical error of about 13% of the price". Remember it is a root-mean-square typical error: individual houses can be far off (large expensive houses are the usual culprits), and predictions converted back with `expm1` are close to median prices, not expected prices.

---

### 2. K-fold cross-validation

#### 2.1 What we estimate and what is biased

We want the expected error of the **procedure** (preprocessing plus model plus hyper-parameters) on a house it has never seen. K-fold cross-validation splits the n development rows into k parts, trains on k-1 parts, scores on the held-out part, and averages the k scores:

CV = (1/k) * sum_j RMSE_j.

Two effects control its quality.

- **Bias.** Each model is trained on (k-1)/n * n rows, fewer than the full data. Models trained on less data are slightly worse, so CV is mildly **pessimistic**. With n = 1,241 development rows and k = 5, each model trains on about 993 rows and is scored on about 248. Larger k lowers this bias (10-fold trains on about 1,117 rows) at the cost of more compute.
- **Variance.** Training sets of different folds overlap heavily (any two share 3/4 of the data when k = 5), so the k scores are positively correlated. That makes the average less stable than an average of k independent numbers would be, and larger k makes the overlap, and the correlation, larger.

Five folds is the standard compromise for 1,241 rows: low enough cost to run hundreds of configurations, bias small compared with differences between models, and enough folds to estimate a spread (decision D2). A single 80/20 split would have much higher variance.

#### 2.2 Why shuffled with a fixed seed

`validation.make_cv` returns `KFold(n_splits=5, shuffle=True, random_state=42)`. Shuffling matters because the file is ordered by `Id`, and Ids can correlate with neighbourhood or sale date; unshuffled folds would each contain a different slice of the market. The fixed seed (`config.SEED`, 42) matters for a different reason: every configuration is scored on **exactly the same five partitions**. That makes comparisons paired (section 2.4) and makes every score reproducible.

#### 2.3 Fold standard deviation versus standard error of the mean

Take E003 (Ridge on base): the five fold scores are 0.120, 0.185, 0.163, 0.112, 0.125. The fold standard deviation is s = 0.0316 (`CVResult.std`, with ddof = 1). It describes how much difficulty varies between folds: fold 2 contains hard houses. The **standard error of the mean** would be s / sqrt(k) = 0.0316 / 2.24 = 0.0141 if the folds were independent, which they are not (section 2.1), so even that figure understates the uncertainty. Neither quantity is the right ruler for comparing two models, because most of the fold-to-fold swing is shared by both models.

#### 2.4 Paired comparisons and the one-SE rule (D14)

Because both models see the same folds, compute the per-fold difference d_j = score_A,j - score_B,j. Fold difficulty cancels, and only the difference between the models remains. The mean difference is d-bar = (1/k) * sum d_j, and its **paired standard error** is

SE_d = s_d / sqrt(k), with s_d the standard deviation of the d_j.

Example from the outlier study (D15): Ridge with Ids 524 and 1299 dropped scores +0.0044 worse than keep-all with a paired SE of 0.0053, even though the fold standard deviation is about 0.03. The paired ruler is about six times tighter than the unpaired one, which is why pairing is worth the trouble.

The registry's retain/reject/investigate decision (`experiments/feature_ablation/_ablation_common.compare_text`, ruling D14) calls a difference real only if |d-bar| exceeds one paired SE. Be precise about what this is. With k = 5 the statistic d-bar / SE_d follows a t-distribution with 4 degrees of freedom, and a value of 1 corresponds to a two-sided p-value of about 0.37. So the one-SE rule is a **noise filter**, not a significance test: it removes differences that are clearly within noise, but a surviving difference is only weak evidence. That is exactly why D14 labels all such decisions provisional and the finalists are re-checked with RepeatedKFold 5x3 on fresh folds (section 3).

---

### 3. Selection bias: the optimism of picking the best

#### 3.1 The mechanism

If you score m configurations on the same folds and keep the one with the lowest CV, you pick the one that was both good **and lucky on those particular partitions**. The minimum of m noisy estimates is lower, on average, than the true error of the winning configuration. The more configurations you compare (a 136-call linear grid, 25 Optuna trials per booster) and the closer they are in true quality, the larger the optimism. The CV score of the winner is no longer an unbiased estimate of its future error.

#### 3.2 What Agent D measured

| Measurement | Result |
|---|---|
| Ridge, 15-point alpha grid selected on the five selection folds | scored 0.1319 there and about 0.1340 on five fresh partitions, so optimism of about +0.002 |
| Lasso E056 | registered 0.1255 on the seed-42 folds, 0.1282 on the seed-7 folds, so +0.0027 |
| Seed-to-seed standard deviation of a CV mean | 0.0026 |
| Boosters, best of 25 trials | not measured; expected +0.002 to +0.005 |

Read this as three facts. Selection optimism is real and about 0.002 to 0.003 for the linear family. A CV mean moves by about 0.0026 simply by changing the partition seed. And therefore two configurations whose scores differ by less than about 0.003 are indistinguishable.

#### 3.3 The rules that follow

1. **Differences below 0.003 between tuned models are not wins** (VALIDATION_AUDIT round 4; EXPERIMENT_HISTORY). A 0.0004 gain from a tweak is noise.
2. **Finalists are re-checked with a fresh seed** (D17): `validation.make_cv(5, 3, config.CV_SEED_FINALISTS)`, with `CV_SEED_FINALISTS = 7`. Finding F20 showed why: `RepeatedKFold(random_state=42)` has as its first repeat the identical five folds used for selection, so a "5x3" check with seed 42 would inherit the selection optimism in one third of its folds. With seed 7 none of the 15 folds equals a seed-42 fold (the largest validation overlap is 25% against 20% expected by chance).
3. **Registry decisions after selection are provisional** (F23): every best-of-grid cell is auto-labelled "retain", to be read as "retain pending the finalist check".
4. **The holdout gives one last, independent estimate** (section 4.3).

---

### 4. Leakage: a taxonomy and where each guard lives

**Leakage** means information from the data you evaluate on influences the model you evaluate. The score then promises more than the model can deliver on new houses. The table lists the kinds that matter here and the guard in this code base.

| Kind | How it would leak | Guard in this project |
|---|---|---|
| Preprocessing | median, scaler, one-hot categories, skew selection computed on all rows | everything fitted lives inside a scikit-learn `Pipeline`; `validation.cross_validate` fits a fresh `clone` of the whole pipeline on each training fold |
| Target | `SalePrice` used to build inputs | cleaning and features never read the target; a test inspects the code for `TARGET` (`tests/test_cleaning.py`) |
| Row filters | outliers removed before the split, so validation loses its hardest rows | `validation.iter_folds` applies the `train_row_filter` after the split, to the training part only (D3, D15) |
| Selection | best of many configurations judged on the folds that picked it | fresh seed for finalists (D17), the 0.003 rule, the single-use holdout |
| Holdout reuse | scoring the holdout repeatedly while tuning | `validation.score_holdout_once` and `HoldoutAlreadyUsedError`, usage log, `_assert_is_holdout_frame` (F21) |
| Stacking | meta-learner trained on in-sample base predictions | nested stacking inside `fit` (cross-reference part 3) |

#### 4.1 Preprocessing inside the pipeline

Section 3 of part 1 covers the principle. The code-level guard is that the model builders (`houseprices.models:build_ridge` and the others) return a complete `Pipeline` that accepts the cleaned DataFrame, and `validation.cross_validate` calls `clone` on it per fold. `LazyPreprocessor` builds the column transformer from the training fold it receives and reads only names and dtypes when doing so. Nothing fitted survives from one fold to the next.

#### 4.2 Training-row filters: only on the training side, and the fold-2 story

Dropping outliers is a modelling choice and must be evaluated like one. `validation.iter_folds` first creates the fold indices on the **full** development frame, then passes only the training rows through the filter (`keep_all`, `drop_candidate_outliers` for Ids 524 and 1299, or `drop_large_living_area(4000)`). The validation fold is never filtered, and `cross_validate` raises if a filter empties a training fold (F3, `ValueError`).

The result (D15, experiments E049 to E054): keep-all won for all three models. The mechanism is visible in the folds. Ids 1299, 692 and 1183 fall in validation fold 1 and Id 524 in fold 2. A model trained without them is still graded on them, so the linear fold standard deviation rises from about 0.023 to about 0.035. A public notebook that deletes the rows before splitting removes them from the validation side as well, deleting the hardest cases from the exam, so its score is better than the model deserves.

#### 4.3 The single-use holdout

`validation.split_development_holdout` carves a 15% holdout (219 houses, stratified by price decile with `StratifiedShuffleSplit`, seed 42) and stores the ids once in `experiments/holdout_ids.json`. `load_or_create_holdout_ids` reads that file as the source of truth and never recomputes it. The development set is the remaining 1,241 houses; model selection never sees the holdout.

At the end, `validation.score_holdout_once(fitted_estimator, X_holdout, y_holdout_log, frozen_experiment_id)` scores it once and appends the event to `experiments/holdout_usage.log`. A second call raises `HoldoutAlreadyUsedError` unless `force=True`, and a forced call is logged as forced so the report must state that the holdout is no longer unbiased. Finding F21 showed that the first version of this guard was advisory, since it accepted any frame, so a mistaken call with development rows would spend the single use. The fix, `_assert_is_holdout_frame`, runs first and refuses a frame whose Id set is not the registered holdout; it is raised before the use is consumed.

#### 4.4 Nested stacking

A stacking ensemble trains a meta-learner on base-model predictions. If those predictions are in-sample, the meta-learner learns to trust whichever base model memorised the training data. `NestedStackingRegressor` avoids this by building out-of-fold predictions **inside** `fit`, so `validation.cross_validate` remains the single outer protocol. Part 3 of the guide covers it; here, remember only that the leakage guard is structural (nesting inside `fit`) and that the test `test_inner_oof_rows_are_never_seen` pins it.

#### 4.5 A leakage you can create after the experiments

`train.py` can fit on `rows="all"`, which includes the holdout. An artefact fitted that way must never be given to `score_holdout_once`. Finding F24 led to a guard: `train.fit_registered` refuses `rows="all"` until `validation.holdout_usage()` shows the holdout has been scored, and the artefact's sidecar records `includes_holdout_rows`.

---

### 5. Reproducibility

#### 5.1 The registry record

Every experiment is one JSON line in `experiments/registry.jsonl`, written by `registry.append`, which assigns the next id (`next_id`, E001, E002, ...) when the id is empty. The `Experiment` dataclass stores identity (id, agent, title), the research protocol (hypothesis, rationale, implementation, evaluation text), the **reproducibility fields** (`builder` as "module:function", `builder_kwargs`, `feature_set`, `train_row_filter`, `cv_n_splits`, `cv_n_repeats`, `seed`), the results (`fold_scores`, `cv_mean`, `cv_std`, `fit_seconds`) and the conclusions (`interpretation`, `decision`, `tags`, `notes`, `timestamp`). `registry.write_summary` regenerates `experiments/registry_summary.csv`, sorted by `cv_mean`.

#### 5.2 The config hash

`Experiment.config_hash` is the first 12 hexadecimal characters of the SHA-256 of a JSON document with sorted keys containing `builder`, `builder_kwargs`, `feature_set`, `train_row_filter`, `cv_n_splits`, `cv_n_repeats` and `seed`. It excludes titles and results. Two experiments with the same hash are the same computation, so a runner can skip a configuration already registered (`_common.find_registered`). A subtlety found later (F26): `find_registered` originally built its probe with the default seed 42, so it could not match a record registered with seed 7, and re-running a finalist created a duplicate. It now takes a `seed` argument. The lesson: a hash is only as useful as the fields that go into it, and every lookup must fill all of them.

#### 5.3 Reproducing an experiment

`python -m houseprices.evaluate --experiment E003` calls `evaluate.run_registered`. It reads the record, resolves the builder string with `data.resolve_callable`, rebuilds the estimator from the stored kwargs, rebuilds the data with `data.load_development_data` and the folds with `validation.make_cv` using the stored repeats and seed, runs `cross_validate`, and compares the new fold scores with the registered ones (tolerance 2e-4). The report says `reproduces_registered_scores: true` or false, and prints `n/a` if you override repeats or seed, because then the protocol differs. Example: E042 reproduced 0.12562 against 0.125622 registered. The environment is pinned in `requirements.txt` (pandas 2.3.3 and so on); finding F14 notes that the record does not store library versions or the git commit, so that part of reproducibility rests on the pinned file (the train.py sidecar does record them).

#### 5.4 Why `train.py` and `predict.py` are not CV evidence

CV evidence is a score on rows the model did not train on. `train.py` fits a registered configuration on a chosen set of rows (development rows or all rows) and `predict.py` writes predictions for the competition test file, which has no prices. Neither produces a held-out score. A model fitted on all rows and then "evaluated" on the same rows would only show how well it memorises. The only valid numbers are the CV scores in the registry, the finalist RepeatedKFold with the fresh seed, and the one logged holdout call.

---

### 6. What the audit rounds found, and why independent review matters

The auditor (Agent D) did not write the code it reviewed, so it could test what the authors assumed. The table lists the findings as recorded in `docs/VALIDATION_AUDIT.md`. Severity is the auditor's; status is as read from the code and tests on writing, and the audit file is the source of truth for any later change.

| ID | Severity | Finding | Status |
|---|---|---|---|
| F1 | Medium | `is_single_pass` was always true (`RepeatedKFold` has no `n_splits`), so out-of-fold predictions of repeated CV were overwritten instead of being NaN | fixed |
| F2 | Medium | NaN predictions scored silently; a broken experiment sank to the bottom of the summary | fixed (raises on non-finite) |
| F3 | Low | a row filter that left no training rows was not detected | fixed (raises) |
| F4 | Low | holdout JSON trusted without checking ids against the data | fixed |
| F5 | Low | dictionary spelling key `CmentBd` should be `CemntBd` | fixed |
| F6 to F8 | Info | duplicate ids allowed in an append-only log; import needs `train.csv`; filter name must encode its threshold | noted, F8 closed for integers |
| F9 | High | the feature table had no `Id`, so the outlier filter raised `KeyError` | fixed (`Id` re-attached in `data.py`) |
| F10 | Low | dry-run data could be registered as a real experiment | fixed (raises in dry-run mode) |
| F11 | Low | text-column detection broke under pandas 3 | fixed |
| F12 to F15 | Info | non-integer filter thresholds, silent fallbacks in LotFrontage options, no library versions in the record, flags chosen by inspecting training data | noted |
| F16 | Low | a stacking test did not detect the classic in-sample meta-learner leak | test strengthened |
| F17 | Low | optimistic diagnostics could be hand-entered into the registry | guard added |
| F18, F19 | Info | fixed blend weights chosen from the same scores are optimistic; `passthrough` unreachable | noted |
| F20 | Medium | the finalist 5x3 check shared its first repeat with the selection folds | fixed (D17, seed 7) |
| F21 | Low | `score_holdout_once` accepted any frame | fixed (`_assert_is_holdout_frame`) |
| F22, F23 | Info | tuning cap not enforced in code; best-of cells auto-labelled retain | noted |
| F24 | Medium | `train.py --rows all` allowed before the holdout is scored | fixed in code (`train.fit_registered` refuses `rows="all"` until the holdout is scored; test `test_fit_rows_all_is_refused_until_the_holdout_is_scored`) |
| F25 | Low | `predict.py` accepts overflowing log predictions that become `inf` prices | fixed in code (`predict_test` now checks that prices are finite) |
| F26 | Low | finalist duplicate detection can never match a seed-7 record | fixed in code (`find_registered` takes a `seed` argument and `run_finalists.py` passes it) |
| F27, F28 | Info | registry append is not atomic or locked; artefacts are pickles | noted |

The lessons are practical. F9 was a High-severity defect hidden behind a comment that was simply false ("X includes Id"). F1 passed every casual read because the code looked right. F20 is a statistical, not a coding, defect: the code did exactly what it said, and the protocol was still subtly biased. F21 shows that a guard that does not check what it guards is advisory. The common pattern is that authors read what they meant, while a reviewer runs what is written. This project's rule is therefore: every binding protocol rule gets an executable test written by someone other than its author, and a finding that is not yet fixed is pinned by a strict expected-failure test so that fixing it forces the marker to be removed.

---

### Self-check questions

1. Show that RMSE on log1p of the price equals the Kaggle RMSLE, and say which function in the code computes it.
2. A model's RMSE is 0.12. At the median price of $163,000, roughly what are the typical high-side and low-side errors? Why are they not symmetric?
3. Why is K-fold CV mildly pessimistic, and why are the k fold scores correlated?
4. Why do we shuffle the folds, and why do we fix the seed?
5. The fold standard deviation of E003 is 0.0316. Why is that the wrong number to judge a 0.003 difference between two models, and what do we use instead?
6. State the one-SE rule precisely and say honestly what it is and is not statistically.
7. Explain selection bias. What did Agent D measure for the Ridge grid and for Lasso E056, and what rule follows?
8. Why does `make_cv(5, 3, 42)` not give an independent finalist check, and what was done about it?
9. Explain where in the code a training-row filter is applied, and why dropping Ids 524 and 1299 before the split would flatter the score. What did the honest protocol show?
10. What does the config hash contain, what does `evaluate --experiment E003` do, and why does a `train.py` fit not count as validation evidence?

### Answers

1. The competition computes sqrt of the mean of (log(1+p) - log(1+a))^2. Training on y = log1p(price) and predicting yhat = log1p(p) makes this exactly sqrt of the mean of (yhat - y)^2, the ordinary RMSE on the log scale. `validation.rmse_log` computes it, and `rmsle_dollars` computes it from dollar values as a check.
2. About 12.7% too high or 11.3% too low, so roughly +$20,800 or -$18,400 at $163,000. The difference of logs is the log of a ratio, and the logarithm is asymmetric: e^0.12 = 1.1275 but e^-0.12 = 0.8869.
3. Each model trains on only (k-1)/k of the data, so it is a little worse than a model trained on everything, which makes CV pessimistic. The training sets of different folds overlap by (k-2)/(k-1) of their rows, so the fold errors are positively correlated and the mean is less stable than an average of independent scores.
4. The file is ordered by Id, which can correlate with neighbourhood or date, so shuffling prevents folds from being different slices of the market. A fixed seed makes every configuration use identical partitions, which allows paired comparison and exact reproduction.
5. Most of the fold-to-fold variation (hard folds versus easy folds) affects every model equally, so it hides nothing about the difference between two models. We compute the per-fold differences, take their mean and the paired standard error s_d divided by the square root of k, which is several times smaller (for example 0.0053 in the outlier study).
6. A difference is called real if the absolute mean per-fold difference exceeds its paired standard error. With 5 folds that is a t-statistic of 1 on 4 degrees of freedom, a two-sided p-value of about 0.37. So it is a noise filter that discards differences clearly within noise, not a significance test, which is why decisions are provisional (D14).
7. Choosing the best of many configurations on the same folds picks one that was also lucky on those partitions, so its CV is optimistic. Ridge's 15-point grid scored 0.1319 on the selection folds and about 0.1340 on fresh ones (+0.002); Lasso E056 scored 0.1255 and 0.1282 on seed-7 folds (+0.0027); the seed-to-seed SD of a CV mean is 0.0026. Hence differences below about 0.003 between tuned models are not wins, and finalists are checked on fresh folds.
8. The first repeat of `RepeatedKFold(random_state=42)` is the identical five folds used for selection, so one third of the check inherits the selection optimism (finding F20). The fix, D17, uses `config.CV_SEED_FINALISTS = 7`; none of the 15 seed-7 folds equals a seed-42 fold.
9. `validation.iter_folds` creates fold indices on the full frame and applies the filter to the training indices only; the validation fold stays complete. Dropping the houses before splitting removes the hardest cases from validation as well, so the exam gets easier. Under the honest protocol, keep-all was best for all three models (D15), and the linear fold standard deviation rose from about 0.023 to about 0.035 when the houses were dropped from training only.
10. The first 12 characters of a SHA-256 over builder, builder kwargs, feature set, row filter, folds, repeats and seed; titles and results are excluded. `evaluate` rebuilds the estimator, data and folds from the record, reruns the CV and compares the fold scores to the registered ones within 2e-4, reporting whether they reproduce. A `train.py` fit produces a model, not a score on unseen rows, and an all-rows artefact has even seen the holdout, so only the registered CV scores, the fresh-seed finalist check and the single logged holdout call are evidence.


---

## Study guide, part 2b: the linear models

Author: feature-researcher. Same audience as parts 1 and 2a. Numbers come from the registry, `docs/EXPERIMENT_HISTORY.md`, the grid file `experiments/models/linear_grid_base+areas+baths+ages+quality_interactions+ordinal_quality_keep_all.csv`, the diversity file `experiments/ensemble/diversity_20261113_121446.csv`, and two documentation measurements made for this chapter (marked as such). All errors are RMSE on log1p(SalePrice), five shuffled folds, seed 42, lower is better.

Notation. X is the design matrix with n rows and p columns, y the log target, beta the coefficient vector. After preprocessing, p = 352 columns for the "all" feature set (53 numeric, about 299 one-hot levels), and each training fold has about 993 rows.

---

### 1. Ordinary least squares and why collinearity breaks it

#### 1.1 The estimator

OLS chooses beta to minimise the sum of squared residuals, ||y - X beta||^2. Setting the gradient to zero gives the normal equations and the closed form

beta_hat = (X'X)^(-1) X'y.

Under the standard model y = X beta + e with noise variance sigma^2, the estimator has covariance

Var(beta_hat) = sigma^2 (X'X)^(-1).

#### 1.2 What collinearity does

Write the eigendecomposition of X'X with eigenvalues lambda_1 >= ... >= lambda_p. Then (X'X)^(-1) has eigenvalues 1/lambda_j. If two columns are nearly proportional (or one is nearly a combination of others), the smallest eigenvalue is close to zero, and the variance of the coefficient along that direction is about sigma^2 / lambda_min, which explodes. The fitted values can stay fine, because the data pin down the *sum* of the collinear effects. But the individual coefficients become huge with opposite signs, and a small change in the training rows swings them wildly. When a validation house has a slightly different mix of the collinear columns, the prediction swings too.

#### 1.3 The Ames data are exactly collinear

This is not a hypothetical. In the base data two exact identities hold on every house (a documentation measurement made for this chapter):

- `GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF`
- `TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF`

With 35 numeric columns plus an intercept, the design matrix has rank 34, not 36, and its condition number is about 7e15, which is numerically singular. The engineered `TotalSF` (basement plus two floors) adds yet another exact dependency. One-hot encoding adds the "dummy variable trap": the dummies of one categorical column sum to one, which equals the intercept column.

Plain OLS (E002, numeric columns only) scores 0.1599 with fold scores 0.134, 0.217, 0.168, 0.136, 0.144 and a fold standard deviation of 0.0348. The 0.217 in fold 2 is what an unstable fit looks like: one fold with a different mix of collinear columns and the hard houses of that fold gives a far larger error than the others. We did not diagnose that fold further, so treat collinearity as the likely, not proven, cause. Compare Ridge on the same folds with far more columns (E003): 0.1409 with fold std 0.0316, and the tuned penalised models reach a fold std of about 0.023. Regularisation does not only improve the mean, it removes instability.

---

### 2. Ridge regression (L2)

#### 2.1 Objective and closed form

Ridge adds a squared-length penalty. In scikit-learn the objective is

minimise ||y - X beta||^2 + alpha ||beta||_2^2

(note: no 1/n in front of the data term). Setting the gradient to zero gives

beta_ridge = (X'X + alpha I)^(-1) X'y.

Adding alpha to the diagonal lifts every eigenvalue of X'X to lambda_j + alpha, so the matrix is always invertible and the variance along the flat direction is bounded by sigma^2 / alpha. Using the singular value decomposition X = U D V', the fitted coefficient along direction j is shrunk by the factor d_j^2 / (d_j^2 + alpha): well-determined directions (large d_j) are barely touched, poorly determined ones (small d_j) are shrunk strongly toward zero. This is exactly the right behaviour for collinear columns: the unstable direction is suppressed and the stable one kept.

Ridge has a bias-variance reading. alpha = 0 is OLS (no bias, high variance). As alpha grows, variance falls and bias rises; the CV error is U-shaped in alpha.

#### 2.2 Shrinkage of one-hot levels

For a one-hot column of a rare category (say 10 houses), the data say little, and the least-squares coefficient is noisy. Ridge shrinks it toward zero, which means "this rare level behaves like the baseline category". Ridge never sets a coefficient exactly to zero: in the fitted E055 pipeline 351 of 352 coefficients are non-zero (a documentation measurement). It can only make small ones smaller.

#### 2.3 The flat alpha curve around 10

From the grid file (feature set "all", skew correction on, RobustScaler):

| alpha | 1 | 3.2 | 5.6 | 10 | 17.8 | 31.6 | 100 |
|---|---|---|---|---|---|---|---|
| cv_mean | 0.13473 | 0.13137 | 0.13064 | 0.13042 | 0.13067 | 0.13134 | 0.13395 |

The minimum is at alpha = 10, and moving alpha by a factor of about 2 in either direction (5.6 or 17.8) costs only 0.0002 to 0.0003. That is well inside the 0.003 noise band (part 2a). The consequence, recorded as D16, is that further tuning of Ridge cannot close its gap: the curve is flat, and the remaining 0.005 difference to the L1 family is structural, not a tuning problem. The alphas of Ridge and Lasso must not be compared directly: Ridge's objective has no 1/n factor, so Ridge's alpha = 10 with about 993 rows corresponds to roughly 10/993 = 0.01 on the scale where Lasso's alpha lives.

---

### 3. Lasso (L1)

#### 3.1 Objective

Scikit-learn's Lasso minimises

(1 / (2n)) ||y - X beta||^2 + alpha ||beta||_1, where ||beta||_1 = sum_j |beta_j|.

There is no closed form, because |beta_j| is not differentiable at zero, but the problem is convex and has a unique solution on well-behaved data.

#### 3.2 Soft-thresholding and sparsity

The simplest case shows the mechanism. If the columns are orthonormal in the sense X'X / n = I, each coefficient decouples and the Lasso solution is the **soft-threshold** of the OLS coefficient:

beta_lasso_j = sign(beta_ols_j) * max(|beta_ols_j| - alpha, 0).

Coefficients smaller than alpha in absolute value are set **exactly to zero**; larger ones are shrunk by alpha. (Ridge, in the same setting and with a penalty alpha ||beta||^2 next to the same 1/(2n) data term, multiplies every coefficient by 1 / (1 + 2 alpha) and zeroes none.) Geometrically, the L1 ball has corners on the axes, so the constrained optimum tends to land on a corner, where some coefficients are zero. This is why Lasso performs variable selection.

For correlated columns the effect is the same in spirit: Lasso keeps one column of a correlated group and zeroes the others, but which one it keeps is somewhat arbitrary and can change between folds.

#### 3.3 How many coefficients survive at the optimum (documentation measurement)

We fitted the E056 pipeline (Lasso, alpha = 5.62e-4, RobustScaler, skew correction, feature set "all") on the 1,241 development rows:

| Quantity | Value |
|---|---|
| Columns after preprocessing | 352 |
| Non-zero coefficients | **115** |
| Zero coefficients | 237 (67%) |
| One-hot columns | 299, of which 82 are non-zero |

So Lasso discards two thirds of the columns, mostly rare category levels. The largest coefficients show a caution you should know. The biggest is `RoofMatl_ClyTile` at -0.60, and in the development rows that level occurs in exactly **one house**; `MSZoning_C (all)` (10 houses) has -0.30 and `Condition2_PosN` (2 houses) -0.17. A coefficient for a single house is simply that house's residual absorbed by a dummy column: it makes the training fit better and is a pure memorisation, which the penalty has not removed because the gain in fit exceeds the penalty at this small alpha. It does not generalise, and it is harmless only because that level is almost never seen again. The meaningful coefficients are those of `Neighborhood` levels (Crawfor, StoneBr, NoRidge, NridgHt, Edwards), `GrLivArea`, `OverallQual`, `TotalSF` and `HouseAge`.

#### 3.4 Why the optimum near 5.6e-4 is sharp

From the grid file (RobustScaler):

| alpha | 1.0e-4 | 3.2e-4 | 5.6e-4 | 1.0e-3 | 1.8e-3 | 3.2e-3 |
|---|---|---|---|---|---|---|
| cv_mean | 0.13397 | 0.12780 | **0.12545** | 0.12650 | 0.13150 | 0.13673 |

Compared with Ridge's flat bowl, Lasso loses 0.0024 by halving alpha and 0.0010 by doubling it (E056 sits at 5.62e-4). The reason is that alpha controls two things at once. It shrinks every surviving coefficient, and it decides *which* coefficients are zero. When alpha is a bit too small, noisy rare levels and redundant columns enter and variance grows. When alpha is too large, genuine signal columns drop below the threshold and are removed, so bias grows quickly. With about 352 mostly weak columns the model sits on a ridge between these two failure modes. Because the optimum is sharp and was found on the same folds that selected it, it carries a selection optimism of about +0.0027 (Agent D's measurement: 0.1255 on the seed-42 folds, 0.1282 on seed-7 folds).

---

### 4. ElasticNet

#### 4.1 Objective

ElasticNet mixes the two penalties with a ratio rho (`l1_ratio`):

(1 / (2n)) ||y - X beta||^2 + alpha * rho * ||beta||_1 + (alpha * (1 - rho) / 2) * ||beta||_2^2.

rho = 1 is the Lasso; rho near 0 is almost Ridge. The L2 part makes the problem strictly convex, so correlated columns tend to share weight (the "grouping effect") instead of one being picked arbitrarily.

#### 4.2 Why l1_ratio between 0.5 and 0.9 ties

The best cv_mean per l1_ratio in the grid is: 0.1 gives 0.12687, 0.3 gives 0.12622, 0.5 gives 0.12555, 0.7 gives 0.12594, 0.9 gives 0.12581, and the pure Lasso gives 0.12545. Between 0.5 and 0.9 the differences are below 0.0004, far inside the noise band. There is a simple reason in the numbers: the registered ElasticNet (E057) has alpha = 1.18e-3 and rho = 0.5, so its effective L1 weight is alpha times rho, about 5.9e-4, which is practically the Lasso optimum of 5.6e-4. The two models are asked to zero out the same columns by the same amount; the extra L2 term, which has a small effect when most columns are one-hot dummies that are not strongly correlated with each other, only smooths the solution. Hence the Lasso and ElasticNet residuals correlate at 0.997 (section 8).

---

### 5. Why skew correction helps linear models by about 0.009 but not trees

The in-fold `SkewLog1pTransformer` (part 1, section 4.5) improved Ridge and ElasticNet by 0.0083 to 0.0094 in the ablation (E039 to E042) and is switched on for all linear models (D12). The reasons are specific to what a linear model assumes.

1. **A linear model assumes a straight line between each input and the (log) price.** For right-skewed inputs the relationship with log-price is usually closer to a straight line in the *logarithm* of the input than in the raw input. (Price per square foot is roughly constant, so log-price is linear in log-area.) After `log1p`, the straight-line assumption is a better fit.
2. **Leverage.** In least squares, observations with extreme input values have large leverage and pull the line toward themselves. `LotArea` has a skewness of 13 in development because of a few giant lots; after `log1p` it is about -0.3. Without the transform a handful of houses control the `LotArea` coefficient.
3. **Penalties depend on scale.** A penalised model charges for coefficient size, and a column dominated by a few huge values has most of its variation in the tails, so its coefficient is determined by those few rows.

**Trees do not benefit** because they split on thresholds: a tree only uses the *order* of values in a column, and `log1p` is strictly increasing, so every split is the same before and after. A transform that does not change order cannot change a tree, so applying it would only add cost (D12).

---

### 6. RobustScaler versus StandardScaler

StandardScaler subtracts the mean and divides by the standard deviation. RobustScaler subtracts the median and divides by the interquartile range (the 25th to 75th percentile), so extreme values do not distort it. Scaling matters for penalised models, since the penalty treats all coefficients equally and the units of a column change its coefficient: a column in square feet needs a tiny coefficient and a column in a 0/1 flag a large one.

The grid shows it **does not matter here**: for every family the best robust and standard scores differ by at most 0.0001 (Ridge 0.13042 vs 0.13028; Lasso 0.12545 vs 0.12556; ElasticNet 0.12555 vs 0.12556 at l1_ratio 0.5). The reasons: after the skew transform most numeric columns are close to symmetric, so mean and median, and standard deviation and IQR, are similar. For the 0/1 dummies the two scalers differ in principle (RobustScaler leaves a dummy whose IQR is 0 unscaled, StandardScaler divides by sqrt(p(1-p)), making rare dummies larger and therefore cheaper to penalise), but those columns are a minority of the signal. We register Ridge with the standard scaler and the L1 models with the robust one only because those were the grid winners; the choice is not a finding.

---

### 7. Convergence, `max_iter` and cost

#### 7.1 Coordinate descent

Lasso and ElasticNet are solved by **coordinate descent**: cycle through the columns, and for each j update beta_j in closed form while holding the others fixed. For the Lasso the update is a soft-threshold:

beta_j <- S( rho_j, alpha ) / z_j, with rho_j = (1/n) x_j' r_j, z_j = (1/n) ||x_j||^2,

where r_j is the residual with the contribution of column j added back. One sweep over all columns costs O(n * p), here about 993 * 352, roughly 350,000 multiply-adds, a fraction of a millisecond. The algorithm stops when the *duality gap* (a certified bound on the distance from the optimum) falls below `tol` times the scale of the target, or when `max_iter` sweeps are reached.

#### 7.2 max_iter and the one warning

The model builders in `houseprices.models` set `max_iter = 20000` (scikit-learn's default is 1000). Convergence is slowest when alpha is small and columns are strongly correlated, because the penalty then barely constrains the problem and coordinate descent zig-zags along flat valleys; each sweep makes only a small step. Of the 136 grid fits, exactly **one** raised a `ConvergenceWarning`: ElasticNet with alpha 1e-4 and l1_ratio 0.1 under the standard scaler, that is, the weakest regularisation with the least L1 in the grid. It is far from the optimum (E057 uses an alpha about 12 times larger) and therefore has no effect on any registered result. If you see a ConvergenceWarning near the optimum, the first fixes are a larger `max_iter` or a larger alpha. Ridge has no such issue, since it is solved in closed form (or by a direct linear-algebra solver).

#### 7.3 Cost in practice

Each five-fold CV of a linear model in this project takes about one second, which is why 136 grid points ran in a few minutes. It is the reason the linear family can be searched exhaustively while the boosters needed Optuna.

---

### 8. What the grid results say (E055 to E057)

| Id | Model | Hyper-parameters | Scaler | cv_mean | Fold std |
|---|---|---|---|---|---|
| E055 | Ridge | alpha = 10 | standard | 0.13028 | 0.0232 |
| E056 | Lasso | alpha = 5.62e-4 | robust | 0.12545 | 0.0235 |
| E057 | ElasticNet | alpha = 1.18e-3, l1_ratio = 0.5 | robust | 0.12555 | 0.0230 |

Reading, with the necessary caution:

- The L1 family beats Ridge by about 0.005 in **every** fold, which is the strongest kind of evidence available with five folds (a consistent paired difference). The explanation offered in the history file is that with about 350 mostly one-hot columns and 993 training rows, zeroing rare levels works better than shrinking them. Our fitted Lasso keeps 115 of 352 columns.
- Lasso and ElasticNet tie (0.0001 apart). Pick either; D16 keeps both as finalists but they are nearly the same model.
- Fold 2, which holds Id 524, remains the hardest fold (0.155 to 0.160) for every linear model.
- The registered bests are the minima of 136 comparisons on the same folds, so they are optimistic by about 0.002 to 0.003 (part 2a). The finalist check on fresh seed-7 folds will state the honest number.

---

### 9. Linear models as ensemble partners

A linear model and a boosted tree make different mistakes: the linear model is smooth and additive, the tree is piecewise constant and finds interactions. Part 3 covers stacking and blending; the evidence for diversity is in the diversity file (`experiments/ensemble/diversity_20261113_121446.csv`, single-pass out-of-fold predictions, used only for this diagnostic).

| Pair type | Residual correlation | Prediction correlation |
|---|---|---|
| linear with linear (Ridge, Lasso, ElasticNet) | 0.970 to 0.997 | 0.996 to 1.000 |
| tree with tree (XGBoost, LightGBM, CatBoost) | 0.946 to 0.962 | 0.994 to 0.996 |
| linear with tree | **0.823 to 0.862** | 0.980 to 0.985 |

The maths of why this matters: take two predictors whose errors have equal variance s^2 and correlation rho. The error of their equal average has variance s^2 (1 + rho) / 2, so its RMSE is s * sqrt((1 + rho) / 2). For rho = 0.85 (linear with tree) that is 0.961 s, about 4% better than either model; for rho = 0.97 (two linear models) it is 0.992 s, under 1%. A third near-duplicate adds almost nothing. This is why the observed gain comes from one linear plus one tree, not from many linear models: the first registered ensemble, an equal average of Lasso and CatBoost (E061, Stage 1 bases), scored 0.11807 against 0.11946 for CatBoost alone (-0.0014, inside the 0.003 noise band; with the final Stage 2 CatBoost the gain shrank to -0.0007, and on the fresh finalist folds the average was 0.0010 better than CatBoost alone, a third of a standard error, see parts 3 and 4), while a blend of the three linear models (E064) was no better than Lasso alone.

Note that the prediction correlations are all 0.98 or higher because the models predict the same target; it is the correlation of the *errors* that decides how much averaging helps.

---

### Self-check questions

1. Write the OLS estimator and its covariance, and explain why exact collinearity makes it unstable. Which two exact identities hold in the Ames numeric columns?
2. Derive the Ridge solution and explain, using the SVD, why it fixes collinearity. Why does Ridge never produce an exact zero?
3. State the Lasso objective as scikit-learn writes it, and why Ridge's alpha = 10 and Lasso's alpha = 5.6e-4 cannot be compared directly.
4. Explain soft-thresholding and why it produces sparsity. How many of E056's 352 coefficients are non-zero, and what is the lesson of the `RoofMatl_ClyTile` coefficient?
5. Why is Ridge's alpha curve flat around 10 while Lasso's optimum near 5.6e-4 is sharp?
6. Define the ElasticNet penalty. Why do l1_ratios from 0.5 to 0.9 tie here, and what is the effective L1 weight of E057?
7. Give two reasons why `log1p` of skewed inputs helps a linear model and why it cannot help XGBoost.
8. RobustScaler versus StandardScaler: what does each do, and why did the choice not matter here?
9. How does coordinate descent work, what does one sweep cost, why was `max_iter` set to 20000, and which grid fit warned?
10. Residual correlation between a linear model and a tree is about 0.85; between two linear models about 0.97. What does this imply for an equal-weight average, and why do three linear models add little?

### Answers

1. beta_hat = (X'X)^(-1) X'y with covariance sigma^2 (X'X)^(-1). With collinear columns X'X has a near-zero eigenvalue lambda_min and the covariance has a term of size sigma^2 / lambda_min, so some coefficients have huge variance and fitted weights swing between folds, although the fitted values can stay fine. In Ames, GrLivArea equals 1stFlrSF plus 2ndFlrSF plus LowQualFinSF, and TotalBsmtSF equals BsmtFinSF1 plus BsmtFinSF2 plus BsmtUnfSF, on every house.
2. Minimising ||y - X beta||^2 + alpha ||beta||^2 gives beta = (X'X + alpha I)^(-1) X'y. In SVD terms the coefficient along direction j is multiplied by d_j^2 / (d_j^2 + alpha), so poorly determined directions (small d_j) are suppressed most and the matrix is always invertible. The penalty is smooth, so the minimiser has no reason to land exactly on zero; 351 of 352 coefficients stay non-zero.
3. It is (1/(2n)) ||y - X beta||^2 + alpha ||beta||_1. Ridge's objective has no 1/n, so with about 993 rows Ridge's alpha = 10 is comparable to roughly 0.01 on the Lasso scale. The alphas are on different scales and compare only within a family.
4. With orthonormal columns, beta_lasso = sign(beta_ols) * max(|beta_ols| - alpha, 0), so every coefficient smaller than alpha in magnitude becomes exactly zero. E056 keeps 115 of 352 coefficients. `RoofMatl_ClyTile` has -0.60 but occurs in one development house: a dummy that absorbs one house's residual is memorisation that does not generalise, so the largest coefficients are not necessarily the most meaningful.
5. Ridge shrinks smoothly and never selects, so changing alpha by a factor of 2 changes the fit gradually (0.0002 to 0.0003 here). Lasso's alpha both shrinks and decides which columns are zero, so a smaller alpha admits noisy levels and a larger one deletes true signal columns, producing a V shape (0.0024 worse at half alpha, 0.0010 at double).
6. The penalty is alpha * rho * ||beta||_1 + alpha * (1 - rho) / 2 * ||beta||_2^2. E057 uses alpha 1.18e-3 and rho 0.5, so its L1 weight is about 5.9e-4, essentially the Lasso optimum of 5.6e-4, and the extra L2 term has little effect on mostly one-hot, weakly correlated columns, so all ratios from 0.5 to 0.9 land at nearly the same point.
7. A linear model assumes straight-line effects and is pulled by high-leverage points: log-price is closer to linear in the log of an area, and a few giant lots no longer dominate a coefficient (LotArea skewness 13 becomes about -0.3). A tree splits on thresholds and log1p is strictly increasing, so every split is unchanged and the transform can only add cost.
8. RobustScaler centres on the median and divides by the interquartile range; StandardScaler uses the mean and standard deviation. After the skew correction most numeric columns are roughly symmetric, so the two scalings are similar, and the 0/1 dummies carry only a minority of the signal. Best scores differ by at most 0.0001 per family.
9. It updates one coefficient at a time in closed form (a soft-threshold for the L1 part) while holding the others fixed, and cycles until the duality gap is below tolerance. One sweep costs O(n * p), about 350,000 operations here. `max_iter` is 20000 because convergence is slow when alpha is small and columns are correlated; one of 136 fits warned: ElasticNet at alpha 1e-4 with l1_ratio 0.1, standard scaler, far from the optimum.
10. For two equal-variance errors with correlation rho, the equal average has RMSE s * sqrt((1 + rho) / 2): about 0.961 s at rho 0.85 (roughly 4% better) and 0.992 s at rho 0.97 (under 1%). A third near-duplicate linear model adds almost nothing, so the gain comes from pairing a linear model with a tree, as E061 (Lasso plus CatBoost) suggests, subject to the 0.003 noise band and the finalist check.


---

## Study guide, part 2c: boosted trees and Optuna tuning

Author: model-researcher. Same audience as parts 1 to 2b. All errors are RMSE on log1p(SalePrice), five shuffled folds (seed 42), 1,241 development rows, about 993 training rows per fold, lower is better. Numbers come from `docs/EXPERIMENT_HISTORY.md`, `experiments/tuning/stage1_summary.json`, the per-trial CSVs `experiments/tuning/stage1_*_trials.csv` (only trials 0 to 24 are Stage 1; the files are rewritten when Stage 2 ends), the source files `src/houseprices/models.py` and `experiments/tuning/run_optuna_stage1.py`, and three small checks run for this chapter (marked "documentation measurement"). The toy examples with five houses are invented for illustration and are labelled as such.

---

### 1. Decision trees for regression

#### 1.1 What a tree is and how it splits

A regression tree partitions the rows into boxes by asking threshold questions ("OverallQual <= 5?") and predicts the **mean of y** inside each box (each leaf). To choose a split the tree tries every feature j and every threshold t between consecutive sorted values, and keeps the pair that minimises the total sum of squared errors of the two children:

SSE(j, t) = sum over i in left of (y_i - mean_L)^2 + sum over i in right of (y_i - mean_R)^2.

Equivalently it maximises the reduction in variance. The same rule is then applied recursively inside each child until a stopping rule (maximum depth, minimum rows per leaf, minimum gain) is hit.

Toy example (invented numbers). Five houses with log prices y = 11.8, 11.9, 12.0, 12.6, 12.7 and OverallQual x = 4, 5, 5, 8, 9. The mean is 12.2 and the parent SSE is 0.16 + 0.09 + 0.04 + 0.16 + 0.25 = 0.70. The split "x <= 5" gives left {11.8, 11.9, 12.0} with mean 11.9 and SSE 0.02, and right {12.6, 12.7} with mean 12.65 and SSE 0.005. Total 0.025, a reduction of 0.675. The split "x <= 4" gives left {11.8} (SSE 0) and right {11.9, 12.0, 12.6, 12.7} with mean 12.3 and SSE 0.50, so it is far worse. The algorithm is greedy: it picks the best split now and never looks ahead.

A single deep tree has low bias and very high variance: it can memorise the 993 training rows and a small change in the data changes the tree. Boosting (section 2) fixes this by combining many **shallow** trees.

#### 1.2 Invariance to monotone feature transforms, and ruling D12

Look at what the split search uses: only the **order** of the values in a column. If g is any strictly increasing function (log1p, square root, a rank transform) the sorted order of g(x) equals that of x, so the best partition of the training rows is the same; only the numeric value of the threshold changes (it is placed between the same two neighbours). Hence a tree fitted on log1p(LotArea) predicts exactly like one fitted on LotArea on the training rows. For unseen rows the two can differ only if a value falls between two neighbouring training values on the wrong side of a re-placed threshold, which is negligible. XGBoost and LightGBM first bin each feature into quantile-based buckets, and quantile bins also depend only on the order, so the invariance holds for them too.

This is the basis of ruling **D12**: the in-fold `SkewLog1pTransformer` is applied to Ridge, Lasso and ElasticNet (it gained about 0.009 for them, part 2b section 5) and never to the trees, because for a tree it cannot change the model and would only add computation. The boosters therefore use `TREE_PREPROCESSOR_DEFAULTS = {"scale": False}` in `models.py`: no scaling and no skew transform.

Two cautions so you do not over-generalise. First, invariance applies to **features**, not to the **target**: the target is log1p(SalePrice), and a tree does care about it because the SSE is computed on y itself. Training on the log target is what makes the boosters optimise the same quantity the competition scores. Second, one-hot columns are still needed here: the preprocessor gives the trees 0/1 columns (299 of 352 columns) for categories; only CatBoost with native categoricals (section 3.3) consumes the raw text.

---

### 2. Gradient boosting

#### 2.1 The additive model

Boosting builds the predictor as a sum of small trees:

F_M(x) = F_0 + eta * h_1(x) + eta * h_2(x) + ... + eta * h_M(x),

where F_0 is a constant (the mean of y), each h_m is a shallow tree, and eta is the learning rate (shrinkage). Trees are added one at a time and earlier trees are never changed.

#### 2.2 Functional gradient descent

Choose a differentiable loss L(y, F). Ordinary gradient descent updates parameters in the direction of the negative gradient. Boosting does the same in **function space**: at round m, compute for every training row the negative gradient of the loss with respect to the current prediction,

r_i = - dL(y_i, F)/dF evaluated at F = F_{m-1}(x_i),

then fit a tree h_m to these pseudo-residuals r_i and add eta * h_m. The tree is a cheap, smooth-enough approximation of "move every prediction downhill".

For squared error, L = (1/2)(y - F)^2, the negative gradient is exactly the ordinary residual y_i - F_{m-1}(x_i). So squared-error boosting is literally "fit a tree to the residuals, add a fraction of it, repeat". Our target is already log1p(SalePrice), so squared error on it is the RMSE the project reports.

Continue the toy example. F_0 = 12.2, residuals -0.4, -0.3, -0.2, +0.4, +0.5. A tree with the split "x <= 5" predicts the leaf mean of the residuals: -0.3 on the left and +0.45 on the right. With eta = 0.1 the new predictions are 12.2 + 0.1 * (-0.3) = 12.17 for the first three houses and 12.2 + 0.1 * 0.45 = 12.245 for the other two, so the new residuals are -0.37, -0.27, -0.17, +0.355, +0.455. Each round removes only 10 percent of what the leaf can explain, and the next tree is free to choose a different split on the new residuals.

#### 2.3 Learning rate versus number of trees

For a fixed leaf, the remaining error in that leaf shrinks like (1 - eta)^m. With the tuned CatBoost rate eta = 0.0347 and 1,700 rounds, (1 - 0.0347)^1700 is about e^(-60), so the part of the training error the trees can express is essentially gone. Two consequences. First, a small eta needs a large M to reach the same fit: the product eta * M is roughly the total distance travelled (about 60 for the tuned XGBoost 0.0392 x 1600 and CatBoost 0.0347 x 1700; 16.5 for LightGBM 0.0103 x 1600, a different tree shape). Second, small steps act as regularisation: many tiny corrections average out the noise that one big step would copy. That is why the tuned boosters are "slow" (section 5) and why the number of trees is a tuned hyper-parameter here rather than chosen by early stopping. This project does not use early stopping because it would need a validation set inside each training fold; the whole rounds count is tuned in the outer cross-validation instead.

#### 2.4 Subsampling and column sampling

Stochastic gradient boosting (Friedman) fits each tree on a random fraction `subsample` of the rows, drawn without replacement, and many libraries also sample columns. Both inject randomness so that successive trees are less correlated and the sum has lower variance; they also make each round cheaper. The tuned values are subsample 0.76 (XGBoost), 0.76 (LightGBM) and 0.62 (CatBoost), with column fractions 0.80, 0.89 and `rsm` 0.79. On 993 rows a tree sees about 600 to 750 of them per round, a visible dose of noise that the cross-validation judged helpful.

---

### 3. The three libraries

#### 3.1 XGBoost: second-order objective, gain, regularisation

XGBoost minimises, at round t, a regularised objective

Obj = sum_i l(y_i, yhat_i + f_t(x_i)) + gamma * T + (1/2) * lambda * sum_j w_j^2 (+ alpha * sum_j |w_j|),

where T is the number of leaves and w_j the leaf values. It expands the loss to **second order**: g_i = dl/dyhat and h_i = d^2 l/dyhat^2 on the current prediction. For squared error g_i = yhat_i - y_i (minus the residual) and h_i = 1. For a leaf containing the rows I with sums G = sum g_i and H = sum h_i, the best leaf value and the best objective contribution are

w* = - G / (H + lambda),  contribution = -(1/2) G^2 / (H + lambda) + gamma.

The gain of a candidate split is therefore

Gain = (1/2) * [ G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L + G_R)^2/(H_L + H_R + lambda) ] - gamma,

and a split is kept only if the gain is positive, which is how gamma prunes.

Worked on the toy example with lambda = 1, gamma = 0. Left leaf has three houses with residual sum -0.9, so G_L = 0.9, H_L = 3; right has G_R = -0.9, H_R = 2. Then G_L^2/(H_L + 1) = 0.81/4 = 0.2025 and G_R^2/(H_R + 1) = 0.81/3 = 0.27; the parent term is 0 because G_L + G_R = 0. Gain = 0.5 * (0.2025 + 0.27) = 0.236. The leaf values are w_L = -0.9/4 = -0.225 and w_R = 0.9/3 = +0.30, against the unregularised means -0.30 and +0.45: lambda shrinks each leaf by the factor n / (n + lambda) (3/4 and 2/3), strongly for small leaves, weakly for big ones. With gamma = 0.3 this split would be rejected (0.236 - 0.3 < 0).

The other knobs in the search space follow from this:

- **min_child_weight** is the smallest allowed H in a child. For squared error h_i = 1, so it is simply the minimum number of rows in a leaf. The tuned value 4 means leaves of at least 4 rows.
- **reg_lambda** is the L2 term above; **reg_alpha** adds an L1 term that soft-thresholds G (leaf values with a small |G| become exactly zero). The tuned values are lambda 0.27 and alpha 0.0015, so explicit penalties are light; the model is controlled by depth 3, min_child_weight 4, subsampling and a small learning rate. (alpha 0.0015 is almost the lower bound 0.001 of the log-uniform range, in practice "off".)
- XGBoost grows trees level by level (confirmed from the booster configuration, `grow_policy = depthwise`), so `max_depth` bounds the leaf count at 2^depth: depth 3 gives at most 8 leaves.

Note the clash of names: XGBoost's gamma (split penalty) has nothing to do with TPE's gamma (a quantile, section 4).

#### 3.2 LightGBM: leaf-wise growth, histograms

LightGBM grows trees **leaf-wise** (best-first): at every step it splits whichever current leaf, anywhere in the tree, has the largest gain, until `num_leaves` leaves exist. A level-wise tree wastes splits on leaves that gain little; leaf-wise reaches lower loss for the same number of leaves, but can grow lopsided, deep branches that fit noise. That is why `num_leaves`, not depth, is the main complexity knob (default `max_depth = -1`, unlimited, checked in the installed library), and why `min_child_samples` (rows per leaf, default 20) matters: it stops a lopsided branch from isolating two or three houses. The space allows 5 to 40; the tuned value is 7, below the default of 20.

Training uses **histograms**: each feature is bucketed once into a limited number of bins (255 by default; library default recalled, not checked here), and a split search scans the bins' gradient sums instead of the sorted rows, with the trick that a child's histogram equals its parent's minus its sibling's. On 993 rows with mostly 0/1 columns the speed gain is small and the binning changes nothing material, but it is the reason LightGBM is the fastest of the three (1.7 min for 25 trials).

**`subsample_freq`.** In LightGBM, row subsampling ("bagging") happens only every `subsample_freq` iterations, and the default is 0, which disables it: `subsample = 0.76` alone would be silently ignored. We confirmed the library defaults (`subsample_freq` 0, `subsample` 1.0). The search space therefore fixes `subsample_freq=1` (bag every iteration), with the comment in `models.py` and an `assert kwargs["subsample_freq"] == 1` when registering. XGBoost and CatBoost do not have this trap.

#### 3.3 CatBoost: symmetric trees, ordered statistics

**Symmetric (oblivious) trees.** In a CatBoost tree every node on the same level uses the **same** split (same feature and threshold). A tree of depth d is therefore just d questions and 2^d leaves, indexed by the d answers. Depth 5 gives 32 leaves. This is a strong structural regulariser (few distinct parameters per tree) and makes prediction fast. It is less flexible per tree, which is why CatBoost needs more rounds.

**Ordered boosting.** In ordinary boosting the residual of house i is computed with a model that was itself fitted using house i, so the residuals are slightly too small (a form of target leakage called prediction shift). Ordered boosting draws a random permutation and calculates the gradient of house i from a model trained only on the houses before it in the permutation. Important honesty point: in the installed CatBoost (1.2.10), with 993 rows the default `boosting_type` is **Plain**, not Ordered (documentation measurement: `get_all_params()` after a tiny fit), and the project never sets it. So E060 uses the standard scheme with CatBoost's other features (symmetric trees, `bootstrap_type` MVS, so `subsample` does act, also checked). Ordered target statistics (next paragraph) are a separate mechanism and are used by the native-categorical variant.

**Ordered target statistics for categoricals.** To use a category as a number, a naive "mean of y per category" leaks the row's own target. CatBoost instead takes a random permutation and encodes house i with the statistic computed only from the houses **before** it with the same category:

TS_i = (sum of y over earlier houses with the same level + a * P) / (number of earlier houses + a),

with a prior P (the global mean) and weight a. Toy example: a level whose three houses appear in permutation order with y = 12.0, 12.4, 12.2, prior 12.2, a = 1. House 1 gets (0 + 12.2)/(0 + 1) = 12.2, house 2 gets (12.0 + 12.2)/(1 + 1) = 12.1, house 3 gets (12.0 + 12.4 + 12.2)/(2 + 1) = 12.2. Nobody sees its own label. CatBoost also builds combinations of categorical columns automatically.

**`rsm` and `l2_leaf_reg`.** `rsm` (random subspace method) is the fraction of features considered when choosing each split: CatBoost's column sampling (0.79 tuned). `l2_leaf_reg` is the L2 penalty on leaf values (default 3, range searched 1 to 30, tuned 2.13): the same idea as XGBoost's lambda.

**One-hot versus native categoricals, and E065.** The default pipeline one-hot encodes the text columns into 299 dummies (of 352 columns in total) and gives CatBoost a purely numeric matrix. `build_catboost(native_categoricals=True)` instead keeps the imputed text columns and `CatBoostNative` passes them as `cat_features`. The hope was that ordered target statistics would use categories better than dummies. What was found (E065, `catboost_native`):

- It ran under a 25-minute cap and completed **5 of 25** trials in 28.0 minutes; best CV 0.12212 (fold std 0.0164), against 0.11946 for one-hot CatBoost after 25 trials (E060).
- Because TPE's first 10 trials are random and the sampler is seeded, the first five native trials used **exactly the same hyper-parameters** as the first five one-hot trials, which gives a paired comparison (documentation measurement from the two trial CSVs). Native was worse in all five pairs: +0.0028, +0.0020, +0.0022, +0.0016, +0.0024, mean +0.0022. In fold 1 (index 0) it was worse in every pair, by 0.005 to 0.011.
- Cost: 104 to 629 s per native trial (median 241 s, 1.7 to 10.5 min) against 4 to 42 s (median 15 s) for one-hot; for the same five configurations the native trials were roughly 15 to 24 times slower. (The checkpoint says 5 to 9 min per trial; the measured range is wider and 5.6 min is the average, 27.97 / 5.)

Reading: this is **partial, not conclusive**. Five trials are not a tuned model, native CatBoost may prefer other settings (its best used 800 rounds, depth 6), and the paired gap of 0.0022 is under the project's 0.003 line, so strictly it is "no evidence of a gain and clear evidence of a large cost", not "proved worse". Hypothesis H3 (native is not worth it on 1,241 rows) stands unrefuted and the director's provisional ruling is not to extend the study.

---

### 4. Hyper-parameter tuning with Optuna

#### 4.1 Why not a grid

The XGBoost space has 8 parameters and the full grid would be astronomically large, and most of its volume is useless (for example depth 8 with 500 rounds). Random search covers important dimensions better than a grid but ignores what it has learned. TPE uses the history of trials to decide where to try next.

#### 4.2 What TPE does

TPE (Tree-structured Parzen Estimator) models p(x | y) instead of p(y | x). After n completed trials it sorts them by objective value and splits them at a quantile gamma (not XGBoost's gamma): the best gamma-fraction is the "good" set and the rest the "bad" set. It fits a density l(x) to the hyper-parameters of the good trials and g(x) to the bad ones (Parzen estimators, that is kernel density mixtures). Then p(x | y) is l(x) if y is below the quantile threshold y* and g(x) otherwise, and the expected improvement over y* turns out to be proportional to

(gamma + (1 - gamma) * g(x) / l(x))^(-1),

so **maximising l(x) / g(x)** is the same as maximising expected improvement: choose settings that are common among good trials and rare among bad ones. In practice Optuna draws 24 candidates from l(x) (`n_ei_candidates = 24`) and keeps the one with the highest ratio.

Defaults read from the installed Optuna 5.0.0 source: `n_startup_trials = 10` (the first 10 trials are pure random sampling, since there is nothing to model yet), and the good-set size is `min(ceil(0.1 * n), 25)`: 2 good trials when n = 20, 3 at n = 25, 8 at n = 75. With this version the sampler uses the multivariate variant for single-objective studies by default, so l and g are joint densities over all parameters rather than products of independent ones.

The effect is visible in our data (documentation measurement). The mean objective of trials 10 to 24 is clearly below that of the random trials 0 to 9: XGBoost 0.1254 versus 0.1322, LightGBM 0.1290 versus 0.1340, CatBoost 0.1217 versus 0.1240. The search concentrates in good regions. (XGBoost's overall best, 0.12307, was drawn at random in trial 3; TPE's trial 11 reached 0.1231 as well.)

#### 4.3 Seeded sampler, SQLite, resumability

`optuna.samplers.TPESampler(seed=42)` makes the proposals reproducible, and the study is stored in `experiments/tuning/optuna_stage1.db` (SQLite, git-ignored) with `load_if_exists=True`. The study name encodes the label, feature set and row filter, so each combination has its own history. On every call the script counts the COMPLETE trials and runs only `n_trials - done` more. This is how Stage 2 works: calling the same command with `--n-trials 75` runs 50 more trials on top of the 25 already stored, and the TPE proposals are conditioned on all of them. A subtlety: the seeded random generator is created afresh at each call and its state is not stored, so a resumed study is a valid TPE run but not bit-for-bit what an uninterrupted 75-trial run would have produced (my reasoning from the code, not tested). A time cap (`--time-cap-minutes`) works through a callback that calls `study.stop()` once the soft deadline has passed; the running trial finishes first, which is why the native study took 27.97 minutes under a 25 minute cap.

#### 4.4 Why the objective is the mean of the five selection folds

The objective returned by a trial is `r.mean`: the mean RMSE over the five folds of `KFold(5, shuffle=True, random_state=42)` (`_common.run_cv`, `validation.make_cv`). Averaging over five folds uses every development row for validation exactly once and is much less noisy than one split. Every trial sees the **same** five partitions, so the differences between trials are paired comparisons (the noise of "which houses landed in which fold" cancels), which helps TPE rank configurations. The price is the next subsection.

#### 4.5 Selection optimism

If you take the best of N noisy estimates you keep the luckiest one. Suppose N = 25 configurations were equally good and each score had noise sigma, the expected minimum would be about 1.97 sigma below the truth. The seed-to-seed standard deviation of a five-fold CV mean was measured at about 0.0026 (audit), so the worst case is about 0.005. Real configurations differ and share folds, so the measured optimism is smaller: the audit found +0.0020 for a 15-point Ridge grid and +0.0027 for Lasso E056 (0.1255 on the selection folds against 0.1282 on fresh seed-7 folds); the history records roughly 0.002 to 0.005 for best-of-25 boosters. This gives the standing rule: **differences below 0.003 are not wins**, and every score in the registry before the finalist step is preliminary. D17 is the repair: finalists are re-scored with `RepeatedKFold(5 x 3)` on `CV_SEED_FINALISTS = 7`, partitions that share no fold with the seed-42 ones (validation overlap at most 25 percent against 20 percent by chance), so the estimate for the chosen configuration is not selection-biased.

---

### 5. The staged budget and the Stage 1 results

#### 5.1 The rule

User correction 7 of the approved plan, ruling D19, and the constants at the top of `run_optuna_stage1.py` (`STAGE2_GAP = 0.003`, `STAGE2_WINDOW = 10`, `IMPROVE_TOL = 1e-4`, `TUNING_CAP_HOURS = 3.0`) define the staged budget:

1. **Stage 1:** 25 trials per booster. Cheap and enough to see the landscape.
2. **Stage 2:** continue the same study to 75 trials only for a booster that is within 0.003 of the best model, **or** whose best-so-far still dropped by more than 1e-4 across the last 10 trials.
3. A total tuning cap of 3 hours (recorded in the summary file; the code does not enforce it, audit F22). Used so far: 10.8 min (Stage 1) plus 28.0 min (E065) = 38.8 min.

The point of staging is to spend compute where the curve is still moving. D19 applied it: XGBoost no (gap 0.0036 and flat since trial 4), LightGBM yes (gap 0.0050 but still improving), CatBoost yes (best and still improving), CatBoost native got its own Stage 1.

#### 5.2 The Stage 1 table

| id | model | CV RMSE | fold std | minutes | trial of best | median trial | worst trial | trials within 0.003 of best |
|---|---|---|---|---|---|---|---|---|
| E058 | XGBoost | 0.12307 | 0.0191 | 2.18 | 3 | 0.12641 | 0.15477 | 11 of 25 |
| E059 | LightGBM | 0.12442 | 0.0191 | 1.66 | 24 | 0.12921 | 0.15931 | 5 of 25 |
| E060 | CatBoost | 0.11946 | 0.0170 | 6.93 | 21 | 0.12133 | 0.12975 | 15 of 25 |

(The last four columns are documentation measurements from the trial CSVs, trials 0 to 24.) Best hyper-parameters, from `stage1_summary.json`:

- E058: learning_rate 0.0392, n_estimators 1600, max_depth 3, subsample 0.757, colsample_bytree 0.796, reg_alpha 0.0015, reg_lambda 0.269, min_child_weight 4.
- E059: learning_rate 0.0103, n_estimators 1600, num_leaves 12, subsample 0.759 (subsample_freq 1), colsample_bytree 0.889, reg_alpha 0.143, reg_lambda 0.371, min_child_samples 7.
- E060: learning_rate 0.0347, iterations 1700, depth 5, l2_leaf_reg 2.13, subsample 0.619, rsm 0.788.

#### 5.3 What the numbers say

**Shallow, slow, subsampled.** Eight leaves for XGBoost (depth 3), 12 for LightGBM, 32 for CatBoost (depth 5, symmetric); learning rates 0.01 to 0.04 with 1,600 to 1,700 rounds; row subsampling 0.62 to 0.76. This matches the situation: 993 training rows, a noisy target, mostly low-order interactions. Deep trees would memorise; many small, noisy steps average out. The average XGBoost leaf holds about 90 rows (roughly 750 sampled rows spread over 8 leaves).

**CatBoost is more forgiving.** The standard deviation of the 25 trial scores is 0.0028 for CatBoost against 0.0068 (XGBoost) and 0.0070 (LightGBM); 21 of its 25 trials beat 0.125 (9 of 25 for XGBoost, 2 for LightGBM). Even CatBoost's **median** trial (0.12133) beats the best XGBoost trial (0.12307) and the best LightGBM trial. Symmetric trees and its defaults are strong regularisers, so there is less to tune and much less to lose from a bad draw. That is also why E060 stays ahead after the 0.003 optimism correction: the gap to XGBoost is 0.0036.

**What the search found** (Spearman rank correlation of each parameter with the score over 25 trials, positive meaning larger values are worse; with 25 points only |rho| above about 0.4 is worth mentioning, so this is a hint, not a result). LightGBM: `num_leaves` +0.64 (best five trials have a median of 12 leaves, worst five 24) and `min_child_samples` +0.43; large `reg_alpha` hurt (the worst trials had alpha 2.5 to 8.9). XGBoost: `min_child_weight` +0.46, depth effect weak (+0.15) but the best five have a median depth of 3 and the worst five of 6. CatBoost: `l2_leaf_reg` +0.45 (best five median 2.1, worst five 9.5), `iterations` -0.42 (best five 1,700, worst five 600), depth about 0.

**Search-space edges.** The LightGBM best learning rate 0.0103 sits against the lower bound 0.01, and its `min_child_samples` 7 near the bound 5, so the space may be cutting off a better region (more rounds at a still lower rate would need `n_estimators` above 2000). XGBoost's reg_alpha is also near its floor. I did not change anything: it is a point to know when reading Stage 2.

---

### 6. The real code

#### 6.1 `src/houseprices/models.py`

Threads are capped for every library (the machine has 20 logical cores, 8 are used so other agents can work):

```python
N_THREADS = 8
TREE_PREPROCESSOR_DEFAULTS: dict[str, Any] = {"scale": False}
```

Each builder returns a **complete** Pipeline (`LazyPreprocessor` + model), so cross-validation refits imputers and encoders on each training fold. `_pipeline(model, ..., tree=True)` merges `TREE_PREPROCESSOR_DEFAULTS`. The XGBoost builder shows the pattern (library defaults except seed and threads, then `params` override):

```python
defaults = dict(random_state=SEED, n_jobs=N_THREADS)
model = XGBRegressor(**{**defaults, **params})
return _pipeline(model, preprocessor_kwargs, drop_cols, tree=True)
```

`build_catboost(native_categoricals=True)` switches the preprocessor to `"one_hot": False`, `pandas_output=True`, and wraps CatBoost in `CatBoostNative`, which treats every non-numeric column as `cat_features` (and fills any leftover gap with "Unknown", because CatBoost rejects missing categoricals).

`OPTUNA_SPACES` maps each booster to a function that receives the trial and returns the **complete** parameter dict. Bounds:

| parameter | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| learning_rate | 0.01 to 0.2, log | 0.01 to 0.2, log | 0.01 to 0.2, log |
| rounds | n_estimators 200 to 2000, step 100 | n_estimators 200 to 2000, step 100 | iterations 300 to 2000, step 100 |
| tree shape | max_depth 2 to 8 | num_leaves 7 to 63 | depth 3 to 8 |
| row sampling | subsample 0.5 to 1.0 | subsample 0.5 to 1.0, `subsample_freq=1` fixed | subsample 0.5 to 1.0 |
| column sampling | colsample_bytree 0.5 to 1.0 | colsample_bytree 0.5 to 1.0 | rsm 0.5 to 1.0 |
| L1 / L2 | reg_alpha, reg_lambda 1e-3 to 10, log | reg_alpha, reg_lambda 1e-3 to 10, log | l2_leaf_reg 1 to 30, log |
| leaf size | min_child_weight 1 to 20 | min_child_samples 5 to 40 | (none) |

The log scale is used for quantities that matter multiplicatively (a rate of 0.01 versus 0.02 is as important as 0.1 versus 0.2).

#### 6.2 `experiments/tuning/run_optuna_stage1.py`

The objective keeps the full parameter dict and stores it with the trial:

```python
params = models.OPTUNA_SPACES[name](trial)       # COMPLETE params, incl. fixed ones
kwargs = {**params, **extra}
r = _common.run_cv(builder, kwargs, fs, flt)
trial.set_user_attr("builder_kwargs", json.dumps(kwargs))
...
return r.mean
```

The reason (from the module docstring): `trial.params` would not contain `subsample_freq`, because the LightGBM space fixes it outside the trial. Registering only `trial.params` would give a model without bagging, a different pipeline from the one that was scored. Storing the complete dict means `houseprices.evaluate` can rebuild the identical estimator with `resolve_callable(builder)(**builder_kwargs)`. `register_best` asserts that the kwargs survive a JSON round trip and that the rebuilt pipeline has the same `repr`, skips registration when the config is already in the registry (`find_registered`), and writes the top-3 trials into the notes. `run_study` creates the study with `TPESampler(seed=42)`, `storage=sqlite:///...`, `load_if_exists=True`, and computes `todo = max(0, args.n_trials - done)`.

The Stage 2 decision is a small function:

```python
def stage2_rule(values, reference):
    best = float(np.min(values)); running = np.minimum.accumulate(values)
    ...
    improving = bool(running[-STAGE2_WINDOW - 1] - running[-1] > IMPROVE_TOL)
    gap = best - reference
    return dict(..., within_gap=bool(gap <= STAGE2_GAP),
                stage2_recommended=bool(gap <= STAGE2_GAP or improving))
```

`running` is the best-so-far curve, so "improving" means the best dropped by more than 1e-4 between trial n-11 and trial n-1. The `reference` is the lowest `cv_mean` in the registry or in the stage, whichever is lower.

---

### 7. Verification notes: inconsistencies found while checking the records

1. `stage1_summary.json` mixes two references. The XGBoost, LightGBM and CatBoost entries were computed with reference 0.11946 (E060), for example XGBoost gap 0.12307 - 0.11946 = 0.0036. The `_rule` block and the `catboost_native` entry were rewritten later with reference 0.118066 (E061, an ensemble, registered in between): native gap 0.12212 - 0.11807 = 0.0041. The D19 decisions do not change with either reference (XGBoost would be 0.0050, still above 0.003), but the file is not internally consistent.
2. `stage2_rule` with 10 or fewer completed trials compares the first and last of the running best. For `catboost_native` (5 trials) this gives `improving_last_10 = true` and `stage2_recommended = true`, which is not meaningful.
3. EXPERIMENT_HISTORY states the gap rule as "< 0.003"; the code uses `<=`. Immaterial.
4. The 5 to 9 min per native trial in the checkpoint: measured 1.7 to 10.5 min, mean 5.6.
5. The registry-wide `reference` makes a base booster compete against the ensemble E061. This is conservative (the gap looks larger), not wrong.

---

### 8. Stage 2 results (E066, E067, ruling D21)

Stage 2 continued the CatBoost and LightGBM studies from 25 to 75 trials (D19, cap 60 min, `--stage 2`). XGBoost did not get Stage 2. Numbers below come from `stage1_summary.json`, the two trial CSVs (now 75 rows each), the logs and registry records E066 and E067, plus documentation measurements from the CSVs. The same seeded folds were used as in Stage 1, so the scores are comparable.

#### 8.1 Results

| | Stage 1 (25 trials) | Stage 2 (75 trials) | change | registered | wall time of Stage 2 |
|---|---|---|---|---|---|
| CatBoost | 0.11946 (E060) | **0.11807** (trial 43) | -0.0014 | E066 | 30.0 min |
| LightGBM | 0.12442 (E059) | **0.12173** (trial 35) | -0.0027 | E067 | 3.2 min |

Best parameters.

- E066 CatBoost: learning_rate 0.0167, iterations 2000, depth 6, l2_leaf_reg 1.41, subsample 0.587, rsm 0.813. Fold scores 0.1039, 0.1323, 0.1391, 0.1023, 0.1128; fold std 0.0167.
- E067 LightGBM: learning_rate 0.0169, n_estimators 2000, num_leaves 8, subsample 0.753 (subsample_freq 1), colsample_bytree 0.891, reg_alpha 0.153, reg_lambda 0.533, min_child_samples 5. Fold std 0.0185.

**Did either beat Stage 1 by more than 0.003? No.** CatBoost improved by 0.0014 and LightGBM by 0.0027, both inside the 0.003 noise band, so neither is a win by the project rule. This is hypothesis H1 (Stage 2 gains less than 0.003) confirmed. A second reason for caution: the best of 75 is selected from three times as many trials as the best of 25, so its optimism is, if anything, larger than the +0.002 to +0.003 measured for Stage 1; part of the 0.0014 may be selection, not skill. Both models are carried forward as ensemble bases (D21) because the choice among bases is re-checked on the fresh seed-7 folds anyway. LightGBM still trails CatBoost by 0.0037, a gap that does exceed the band.

**Minutes.** Stage 2 took 33.19 min (30.01 CatBoost, 3.15 LightGBM). With the 38.8 min already used (10.8 Stage 1 plus 28.0 for E065) the total is about 72 min of the 180 min cap.

#### 8.2 How the search behaved

How many of the 50 new trials beat the Stage 1 best: 12 for CatBoost (below 0.119457) and 24 for LightGBM (below 0.124416). The best-so-far curves:

| trial count | 25 | 35 | 45 | 55 to 75 |
|---|---|---|---|---|
| CatBoost best-so-far | 0.11946 | 0.11907 | 0.11807 | 0.11807 (flat) |
| LightGBM best-so-far | 0.12442 | 0.12278 | 0.12173 | 0.12173 (flat) |

Both bests were found by trial 45 and nothing improved over the next 30 trials, so `improving_last_10` is false for both. (`stage1_summary.json` still shows `stage2_recommended = true` for CatBoost, only because its gap to the registry reference 0.118066, the ensemble E061, is about 0.000006. The rule has no Stage 3 and D21 closes tuning.)

**TPE concentrates.** Spread of the objective over blocks of 25 trials (standard deviation, then median):

| block | CatBoost std / median | LightGBM std / median |
|---|---|---|
| trials 0 to 24 | 0.00275 / 0.12133 | 0.00695 / 0.12921 |
| trials 25 to 49 | 0.00061 / 0.11985 | 0.00188 / 0.12514 |
| trials 50 to 74 | 0.00047 / 0.11992 | 0.00160 / 0.12383 |

The CatBoost std fell about six-fold: the last 50 trials all lie between 0.1181 and 0.1210, and 58 of the 75 trials are within 0.003 of the best. That is a plateau, not a peak: many different settings in the same region score the same to within the noise. The shift in the parameters from the first 25 trials to the last 25 (medians): CatBoost learning rate 0.038 to 0.015, iterations 1,500 to 1,900, l2_leaf_reg 2.6 to 1.26, rsm 0.69 to 0.81, depth 5 to 6. LightGBM: num_leaves 15 to 10, min_child_samples 12 to 6, colsample 0.81 to 0.93, subsample 0.79 to 0.72, rounds 1,600 to 1,800. The direction is the same in both: slower, a bit more capacity per tree, weaker penalties, fewer rows per leaf.

#### 8.3 The search-space edges

Both Stage 2 bests sit on the upper bound of the tree-count range (2,000) with a learning rate of 0.017, and the five best LightGBM trials all have `min_child_samples = 5`, the floor of its range (best trial: 8 leaves, 7 is the leaf floor). Counting trials (documentation measurement): iterations or n_estimators at 2,000 in 12 of 75 CatBoost trials and 16 of 75 LightGBM trials (7 each among the last 25); `min_child_samples = 5` in 16 of 75 LightGBM trials (6 of the last 25); learning rate within 25 percent of its floor 0.01 in 9 (CatBoost) and 10 (LightGBM) trials.

What this implies. TPE cannot propose outside the box, so the true optimum along these axes may lie beyond it: with a smaller learning rate it would use more than 2,000 rounds, and with a smaller leaf limit it would allow leaves of fewer than 5 rows. The pattern "lower rate, more rounds" is the same trade described in section 2.3: the product learning rate x rounds is about 33 for both E066 and E067 (0.0167 x 2000 and 0.0169 x 2000), against 59 for E060, so the step budget has been kept but spread over smaller steps. A floor on `min_child_samples` at 5 pushing against its bound means LightGBM wants leaves that nearly isolate single houses, possibly a sign that it is fitting fine structure with little penalty (my interpretation, not tested).

Why stop anyway. First, the returns have fallen sharply: 50 extra trials bought 0.0014 and 0.0027, and the last 30 bought nothing. Second, widening the box would be a new search (new bounds, a new study, a new cost of 2,000+ rounds per trial for CatBoost at about 37 s a CV), and any further gain would be of the order of the gain just seen, i.e. below the 0.003 line and below the selection optimism of 0.002 to 0.003, so it could not be told apart from noise on these five folds. Third, the plateau (58 of 75 CatBoost trials within 0.003 of the best) says the model is insensitive in this region, so the hyper-parameters are not where the remaining error lives; the remaining error is concentrated in a few houses (Ids 1299, 524, 1183, 692; interpretability analysis of E060) and in the cheapest and most expensive deciles. Ruling D21 therefore closes tuning, and the honest statement for an interview is: "both tuned bests touch a boundary of the search space; we know that, we judged the likely gain to be smaller than the noise, and we spent the remaining budget on ensembles and the fresh-fold check instead."

---

### 9. Questions you should be able to answer

1. Why does a regression tree pick its split by SSE, and what does a leaf predict? **A:** It chooses the feature and threshold that minimise the summed squared error of the two children (maximum variance reduction), and a leaf predicts the mean of y of its rows. The search is greedy, one split at a time.
2. Why is `skew_log1p` applied to Lasso but not to XGBoost (D12)? **A:** A tree uses only the ordering of a feature, and log1p is strictly increasing, so the splits are unchanged; for a linear model the transform makes effects closer to linear and removes high-leverage points (gain about 0.009). The target is log-transformed for all models; that is a different matter.
3. Explain gradient boosting for squared error. **A:** Start from the mean, then repeatedly fit a small tree to the current residuals (the negative gradient of the squared loss) and add eta times its prediction. Each step moves the predictions downhill in function space.
4. Why are the tuned learning rates small and the round counts large? **A:** Small eta makes each correction conservative, so noise is averaged out; the remaining error in a leaf falls like (1 - eta)^m, so a small eta needs many rounds (eta x M is about 60 for XGBoost and CatBoost here). Rounds are tuned in the outer CV instead of early stopping.
5. Write the XGBoost split gain and say what lambda, gamma and min_child_weight do. **A:** Gain = 1/2 [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma. lambda shrinks leaf values by n/(n+lambda) for squared error and lowers the gain of small leaves; gamma is the minimum gain that justifies a split; min_child_weight is the minimum sum of h in a child, which for squared error is the minimum row count (tuned 4).
6. Why must `subsample_freq` be 1 for LightGBM? **A:** LightGBM only bags rows every `subsample_freq` iterations and the default is 0 (off), so `subsample` alone is ignored. The space fixes it to 1, and registration asserts it. This is also why the full params, not `trial.params`, are stored.
7. What is leaf-wise growth, and why is `num_leaves` the key knob? **A:** LightGBM always splits the leaf with the largest gain anywhere in the tree, which gives lopsided deep trees; depth no longer bounds complexity, the leaf count does. Tuned: 12 leaves, with `min_child_samples` 7, and `num_leaves` had the strongest rank correlation (+0.64) with the score.
8. What makes CatBoost different, and did native categoricals help? **A:** Symmetric trees (same split per level, 2^depth leaves), ordered target statistics for categoricals (each house encoded from earlier houses only, so no label leak), ordered boosting available as an option (the default here is Plain). E065, five of 25 trials under a time cap: best 0.12212 versus 0.11946 for one-hot; on the five identical configurations native was worse by 0.0016 to 0.0028 and 15 to 24 times slower. Partial evidence, not conclusive.
9. What does TPE do, and what are the defaults? **A:** It splits past trials at the gamma quantile into good and bad sets, models l(x) and g(x), and proposes the candidate with the largest l(x)/g(x) (equivalent to expected improvement). First 10 trials random, good set min(ceil(0.1 n), 25), 24 candidates per proposal. The sampler is seeded and trials are stored in SQLite so a study can resume and grow from 25 to 75.
10. Why is the best-of-25 score optimistic, and how is it handled? **A:** The minimum of noisy estimates is biased low (for equal configurations about 1.97 sigma; the audit measured +0.002 to +0.003). Hence differences below 0.003 are not wins, and finalists are re-scored on fresh seed-7 repeated folds (D17).
11. Why was Stage 2 given to CatBoost and LightGBM but not XGBoost? **A:** The rule is gap to the best model at most 0.003, or best-so-far still dropping over the last 10 trials. XGBoost's gap was 0.0036 and its best came at trial 3 with no improvement afterwards; LightGBM's gap was 0.0050 but its best came at the last trial; CatBoost was the best model and still improving (D19).


---

## Study guide, part 3: ensembles, blending and nested stacking

Code: `src/houseprices/ensemble.py`, `experiments/ensemble/run_ensembles.py`, `tests/test_ensemble.py`.
Scores are RMSE on log1p(SalePrice), which is the Kaggle RMSLE. Lower is better.

### 1. Why ensembles help

A model's expected squared error splits into bias, variance and irreducible noise. Averaging several models does not change the noise and barely changes the shared bias. It reduces the variance part, and only the part that is not common to all members.

For two predictors with equal error variance and error correlation rho, the error variance of their average is sigma^2 (1 + rho) / 2. With rho = 1 there is no gain. With rho = 0 the variance halves. The whole game is therefore low error correlation at similar accuracy. A much worse model rarely helps, because it adds its own bias.

Our models differ in kind. Ridge, Lasso and ElasticNet are penalised linear models on one-hot and scaled inputs. XGBoost, LightGBM and CatBoost are boosted trees that capture thresholds and interactions. Within a family the errors are very similar. Across families they differ more.

#### What the two correlations tell you

| Quantity | Computed on | Reads as |
|---|---|---|
| Prediction correlation | OOF predictions of two models | Almost always above 0.9 here, because every model predicts the same price scale. Nearly useless for judging diversity. |
| Residual correlation | prediction minus truth | The informative one. Near 1 means the models make the same mistakes. Lower values mean blending can cancel errors. |

Stage 1 values (superseded by the final-bases table in section 9.1), from `experiments/ensemble/diversity_20261113_121446.csv`: single-pass OOF predictions of the six tuned Stage 1 bases (E055 to E060) on the working feature set, seed-42 folds. Each base's recomputed cv_mean matches its registered value to six decimals.

Residual correlation (3 decimals):

| | ridge | lasso | enet | xgb | lgbm | cat |
|---|---|---|---|---|---|---|
| ridge | 1.000 | 0.970 | 0.975 | 0.825 | 0.823 | 0.852 |
| lasso | 0.970 | 1.000 | 0.997 | 0.831 | 0.829 | 0.860 |
| elasticnet | 0.975 | 0.997 | 1.000 | 0.833 | 0.834 | 0.862 |
| xgboost | 0.825 | 0.831 | 0.833 | 1.000 | 0.962 | 0.946 |
| lightgbm | 0.823 | 0.829 | 0.834 | 0.962 | 1.000 | 0.946 |
| catboost | 0.852 | 0.860 | 0.862 | 0.946 | 0.946 | 1.000 |

Prediction correlation (3 decimals):

| | ridge | lasso | enet | xgb | lgbm | cat |
|---|---|---|---|---|---|---|
| ridge | 1.000 | 0.996 | 0.997 | 0.980 | 0.980 | 0.983 |
| lasso | 0.996 | 1.000 | 1.000 | 0.982 | 0.981 | 0.985 |
| elasticnet | 0.997 | 1.000 | 1.000 | 0.982 | 0.982 | 0.985 |
| xgboost | 0.980 | 0.982 | 0.982 | 1.000 | 0.996 | 0.994 |
| lightgbm | 0.980 | 0.981 | 0.982 | 0.996 | 1.000 | 0.994 |
| catboost | 0.983 | 0.985 | 0.985 | 0.994 | 0.994 | 1.000 |

Per-base RMSE: ridge 0.1319, lasso 0.1272, elasticnet 0.1272, xgboost 0.1242, lightgbm 0.1256, catboost 0.1204.

Reading:

- Prediction correlation is between 0.980 and 1.000 for every pair, so it says nothing about diversity, as predicted above.
- Within a family the models are near duplicates: Lasso and ElasticNet at 0.997, XGBoost and LightGBM at 0.962, CatBoost with either at 0.946.
- Cross-family pairs are the most complementary, with residual correlation 0.82 to 0.86. The lowest are Ridge with LightGBM (0.823) and Ridge with XGBoost (0.825).
- CatBoost is the most accurate base. Its best partners by diversity are the linear models (0.852 to 0.862), which is why Lasso plus CatBoost is the natural pair.
- Residual correlation is high everywhere, so expect modest ensemble gains.

The diversity report uses single-pass OOF predictions, one per row, from the same folds as the registry. It is a diagnostic. Nothing is fitted on it and then scored on the same rows.

### 2. Averaging versus learned blend weights

An equal-weight average has no parameters, so it cannot overfit. It is the naive reference. A learned blend picks weights w that minimise the squared error of the weighted prediction against the target.

Constraints matter:

- **Non-negative:** without it, least squares may give a model a negative weight and "subtract" it. That usually exploits noise in the training predictions, and it does not generalise.
- **Sum to one:** keeps the blend on the scale of the target and removes a free scaling parameter. The code solves it with a constrained optimiser (SLSQP). With only non-negativity it uses NNLS, non-negative least squares.

The weights are learned inside `fit` from inner out-of-fold predictions of the rows received. The outer validation fold never influences them.

#### Audit F18: weights chosen on the folds you score

If someone picks fixed weights by looking at OOF scores from earlier experiments on the same development folds, the blend's CV score is optimistic. The weights have seen the validation targets indirectly. We therefore report only `weights=None` (learned in fit) or equal weights, which are fixed before looking at any score.

### 3. Stacking with a Ridge meta-learner

Stacking replaces the weighted average with a second model. It takes the base predictions as features and learns the target. We use Ridge with alpha 1 and `passthrough=False`.

- **Ridge:** a low-capacity meta-learner. With six highly correlated columns, an unregularised or flexible meta-model would chase noise. Ridge shrinks and stabilises the coefficients.
- **passthrough=False:** the meta-learner sees only base predictions, not the raw features. Passing raw features back in would give it room to relearn the problem and overfit. It also would not work on our mixed-type frames. This was a director decision, and the code defaults to it.

Unlike a blend, the stack allows weights that do not sum to one and may be negative, plus an intercept. That makes it more flexible and riskier.

### 4. Binding correction 1: nested stacking, one outer fold step by step

The user's correction: the meta-learner must never be scored on rows it was trained on. The classic mistake is to fit bases, collect OOF predictions for the whole development set, fit the meta-learner on them, and report CV on the same rows. The meta-learner has already seen those rows' targets.

Our fix is nested CV. The inner loop lives inside `fit`, so the outer protocol (`validation.cross_validate`) stays the single source of truth, and any ensemble reproduces through `python -m houseprices.evaluate --experiment <id>`.

Worked example, rounded. Development set 1,241 rows. Outer fold 3 of 5 holds out about 248 rows. The outer-train part has about 993 rows. A training-row filter, if any, has already removed outlier rows from these 993 and never touches the 248.

```
DEVELOPMENT (1241 rows)
|-- OUTER VALIDATION, fold 3 (248 rows)  ----- hidden until the very end
|-- OUTER TRAIN (993 rows) -> stack.fit(X_tr, y_tr)
      Step 1  inner 5-fold on the 993 rows (about 199 rows per inner fold)
              for each inner fold i and each base b:
                  fit b on the other 4 inner folds (about 794 rows)
                  predict inner fold i      -> column b of the OOF matrix
              result: a 993 x k matrix; every entry came from a model
              that did not see that row
      Step 2  fit the meta-learner (or blend weights) on that matrix
              with target y_tr
      Step 3  refit every base on all 993 rows
stack.predict(outer validation 248 rows):
      bases (Step 3) predict -> meta-learner (Step 2) combines -> score RMSE
```

The 248 outer validation rows are used exactly once, for scoring. Nothing the meta-learner learned involved them, so the fold score is an honest estimate. The meta-learner in Step 2 learned on out-of-sample base predictions, which look like the predictions the refit bases will make on unseen rows. That match in distribution is the reason to use OOF rather than in-sample predictions: in-sample predictions are too good, so the meta-learner would over-trust the flexible model.

The data flow in the other direction matters too. The meta-learner is trained on Step 1 predictions made by models trained on 80% of the outer-train rows, but applied to predictions from models trained on 100% of them. This small mismatch is standard and harmless.

### 5. Why the "OOF-stack" convenience score is optimistic

`optimistic_oof_stack_score` computes one OOF matrix on the full development set, fits the meta-learner on all of it, and scores the meta-learner on those same rows. It is quick, and it is a plausible-looking number. But the meta-learner was trained on the targets it is scored against. With a few Ridge coefficients the optimism is small. With a flexible meta-model or many bases it can be large. `in_sample_stack_score` is worse: the bases themselves are also scored on their own training rows.

Audit finding F17 asked for a guard against registering these numbers. We did two things in `ensemble.py`:

1. Both functions return an `OptimisticScore`, a float subclass with the attribute `registerable = False`, a `kind` label, and a repr that says "NEVER REGISTER".
2. It has no `fold_scores` or `mean`, so `_common.run_and_register(result=...)` fails with an `AttributeError` before anything is written. It is also always truthy. Otherwise a score of exactly 0.0 would be falsy and make the helper silently run a real CV instead.

The limit of the guard: someone could still build a registry `Experiment` by hand with that float as `cv_mean`. The registry has no check for it, which is why the auditor marked F17 as low. The runner never calls these functions.

### 6. The memoriser leak test and its blind spot (audit F16)

To prove the nested design catches the leak, a test uses a "memoriser" base. It returns the exact target for rows it was trained on and the training mean for any unseen row. The in-sample stack score of a memoriser is near zero (perfect cheating). The nested score must be about as bad as predicting the mean.

The blind spot (F16): with a memoriser as the only base, a leaky stack scores the same as the honest one. For unseen rows the memoriser returns a constant, so whatever weight the meta-learner gave it is irrelevant. The test cannot tell whether the meta-learner was trained on in-sample predictions. Another test (inner OOF rows are never seen by the model that predicts them) does protect the OOF generation, but the meta-training-data leak had no direct test in our file.

Fix: use two bases, a memoriser and a Ridge. In the leaky version the meta-learner sees the memoriser's perfect in-sample predictions, gives it nearly all the weight, and then collapses on unseen rows, because the memoriser returns a constant there. Ridge is the useful base that the honest stack should lean on. Expected pattern per the auditor's probe:

| Variant | Score relative to Ridge alone |
|---|---|
| Honest nested stack | within about 15% |
| Leaky mutant | more than 3 times worse |

The point of the second base is that the leak now has something to damage.

### 7. Nested cost arithmetic

Per outer fold, with k bases and 5 inner folds: 5 x k inner fits plus k refits, so 6k fits. Over 5 outer folds that is 30k fits. A plain 5-fold CV of the same bases costs 5k fits.

| Bases | Plain CV fits | Nested fits per outer fold | Nested fits total |
|---|---|---|---|
| 2 | 10 | 12 | 60 |
| 6 | 30 | 36 | 180 |

Inner fits use about 64% of the rows (80% of 80%). Measured against one fit at 80%, the nested cost is about (5 x 0.64 + 0.8) / 0.8 = 5 times a single 5-fold CV. The runner uses 5.5 as a slightly conservative factor, because fit time is not perfectly linear in rows. Estimate of an ensemble = 5.5 x the sum of its bases' registered CV times. CatBoost dominates the total. The runner prints this estimate, and takes `--budget-minutes` (default 40), skipping experiments that would overrun it.

### 8. How the runner compares an ensemble to its best member

For each ensemble the runner takes the best single member among its own bases (lowest registered cv_mean), and also the best base overall. Both were scored on the same five folds as the ensemble, because the seed and splitter are fixed. So the comparison is paired:

1. d_i = ensemble score minus member score on fold i, for i = 1..5.
2. Mean difference = average of d_i. Negative means the ensemble is better.
3. Paired standard error = std(d_i, ddof 1) / sqrt(5).
4. One-SE rule: mean difference below minus one SE means retain. Above plus one SE means reject. Otherwise investigate.

Why paired: fold difficulty varies a lot (fold scores differ by a few hundredths) and mostly cancels in the differences. Comparing the two cv_means with their unpaired standard deviations would hide real gains. Why only one SE and five folds: the SE from five differences is itself noisy, so a retain means "worth confirming" and not "proven". The director re-tests finalists with RepeatedKFold 5x3.

Decisions are also made against the best member, not against the average of members. An ensemble that beats the average member but not the best one is not worth its extra fitting time.

### 8b. Worked example: average versus learned weights versus six-base blend (E061 to E064)

Stage 1 history. The final-bases rerun (E068 to E075, section 9) and the fresh-fold check (section 10) supersede these numbers.

These four records were registered ahead of the planned full ensemble run, so they are labelled premature in the experiment history. They are valid CV runs. Caveat: they use the Stage 1 tuned bases (E055 to E060), the seed-42 five-fold split and a single repeat. With five folds, a paired SE is noisy. Treat the pattern as a hypothesis for the RepeatedKFold 5x3 test, not as a conclusion.

Reference: best single base is CatBoost E060 at 0.11946. Differences are ensemble minus CatBoost, averaged over the five folds, with the paired standard error.

| Id | Ensemble | cv_mean | Difference vs CatBoost | Paired SE | Runner decision |
|---|---|---|---|---|---|
| E061 | Equal-weight average, Lasso + CatBoost | 0.11807 | -0.0014 | 0.0019 | investigate |
| E062 | Learned blend, Lasso + CatBoost | 0.11902 | -0.0004 | 0.0016 | investigate |
| E063 | Learned blend, all six bases | 0.11943 | -0.00003 | 0.0019 | investigate |
| E064 | Learned blend, Ridge + Lasso + ElasticNet | 0.12585 | n/a | n/a | reject |

E064 is compared with its own best member, Lasso (0.12545), not with CatBoost, and is worse by more than one SE.

What to take from it:

- **Average versus learned weights.** The naive average scored slightly better than the learned blend of the same two models (0.11807 against 0.11902). Their difference is 0.0009 with an SE of 0.0007, so it is within noise. With two highly accurate partners, estimating weights from about 1,000 inner OOF rows adds a little variance that equal weights avoid. This is the usual reason to keep the naive reference.
- **Six bases did not beat two.** Adding Ridge, ElasticNet, XGBoost and LightGBM brought nothing over CatBoost alone. This agrees with the diversity tables: the extra models are near duplicates of Lasso or of the boosted trees.
- **Same-family blends do not help.** The linear-only blend is worse than its own best member, since the three linear models share almost all their errors (0.970 to 0.997).
- **One-SE rule in action.** No ensemble beats CatBoost by more than one SE, so none is "retain". All are "investigate" except the rejected one.

### 9. Final bases: ensembles after Stage 2 (E068 to E075)

Stage 2 tuning replaced CatBoost E060 (0.11946) with E066 (0.11807) and LightGBM E059 (0.12442) with E067 (0.12173). The ensembles were rerun on the six final bases (E055 Ridge, E056 Lasso, E057 ElasticNet, E058 XGBoost, E067 LightGBM, E066 CatBoost), same seed-42 five-fold split, nested protocol as in section 4. Sections 1 and 8b above are kept as the Stage 1 story. Where a number there is superseded, this section gives the new one.

#### 9.1 Diversity with the final bases

Source: `experiments/ensemble/diversity_20261116_234858.csv`. Pooled OOF RMSE (one prediction per row, so slightly above the fold-mean values in the registry) and residual correlation, 3 decimals.

| | ridge | lasso | enet | xgb | lgbm | cat |
|---|---|---|---|---|---|---|
| ridge | 1.000 | 0.970 | 0.975 | 0.825 | 0.827 | 0.846 |
| lasso | 0.970 | 1.000 | 0.997 | 0.831 | 0.837 | 0.856 |
| elasticnet | 0.975 | 0.997 | 1.000 | 0.833 | 0.840 | 0.860 |
| xgboost | 0.825 | 0.831 | 0.833 | 1.000 | 0.972 | 0.939 |
| lightgbm | 0.827 | 0.837 | 0.840 | 0.972 | 1.000 | 0.951 |
| catboost | 0.846 | 0.856 | 0.860 | 0.939 | 0.951 | 1.000 |

| Base | Pooled OOF RMSE | Mean residual corr. with the other five |
|---|---|---|
| ridge | 0.1319 | 0.888 |
| lasso | 0.1272 | 0.898 |
| elasticnet | 0.1272 | 0.901 |
| xgboost | 0.1242 | 0.880 |
| lightgbm | 0.1228 | 0.885 |
| catboost | 0.1190 | 0.890 |

Prediction correlation is again between 0.980 and 1.000 for every pair. The structure is unchanged: linear to linear 0.970 to 0.997, tree to tree 0.939 to 0.972, linear to tree 0.825 to 0.860. Against the Stage 1 table the tuned trees are more accurate (CatBoost 0.1204 to 0.1190, LightGBM 0.1256 to 0.1228) and the CatBoost-Lasso residual correlation moved from 0.860 to 0.856. Diversity barely changed, but CatBoost got better, and that is what matters below.

#### 9.2 Results

Paired difference is ensemble minus its best member on the same five folds (negative is better), with the paired SE. For every ensemble except E074 the best member is CatBoost E066 (0.118072). For E074, three linear models, it is Lasso E056 (0.12545).

| Id | Ensemble | CV mean | Diff vs best member | Paired SE | Diff / SE | Verdict |
|---|---|---|---|---|---|---|
| E068 | average, Lasso + CatBoost | 0.117344 | -0.00073 | 0.00194 | -0.37 | investigate |
| E072 | stack (Ridge meta), Lasso + CatBoost | 0.117479 | -0.00059 | 0.00165 | -0.36 | investigate |
| E069 | NNLS blend, Lasso + CatBoost | 0.118022 | -0.00005 | 0.00174 | -0.03 | investigate |
| E073 | stack, six bases | 0.118464 | +0.00039 | 0.00222 | +0.18 | investigate |
| E075 | stack, three boosters | 0.118582 | +0.00051 | 0.00124 | +0.41 | investigate |
| E071 | NNLS blend, three boosters | 0.118722 | +0.00065 | 0.00094 | +0.69 | investigate |
| E070 | NNLS blend, six bases | 0.118937 | +0.00087 | 0.00226 | +0.38 | investigate |
| E074 | stack, three linear models | 0.125927 | +0.00048 | 0.00035 | +1.37 | reject |

Nothing reaches "retain" (a difference below minus one SE). The best, E068, is at minus 0.37 SE. E074 is worse than its own best member by more than one SE, so it is rejected, as E064 was in Stage 1.

#### 9.3 Reading

- **The gain from Lasso shrank as CatBoost improved.** In Stage 1 the equal-weight Lasso + CatBoost average was 0.11807 against CatBoost 0.11946 (E061, difference -0.0014). With the Stage 2 CatBoost it is 0.11734 against 0.11807 (E068, difference -0.0007). The ensemble improved by 0.0007 while its best member improved by 0.0014, so the margin halved.
- **Six bases and three boosters are slightly worse than CatBoost alone.** E070, E071, E073 and E075 all have positive differences (+0.0004 to +0.0009), each within one SE. Ridge, ElasticNet, XGBoost and LightGBM are near duplicates of Lasso or CatBoost (residual correlation up to 0.997 within the linear family and 0.951 with CatBoost), so they add weight-estimation variance and almost no diversity.
- **Learned weights did not beat equal weights.** For the Lasso + CatBoost pair the order is average (0.11734), stack (0.11748), NNLS blend (0.11802). With two accurate partners and about 1,000 inner OOF rows, estimating weights costs more than it gains.

#### 9.4 Why: the two-model variance formula, with numbers

For unequal members, let Lasso and CatBoost have error standard deviations s1 and s2 and residual correlation rho. With weight w on CatBoost the error variance of the combination is

```
Var(w) = w^2 s2^2 + (1-w)^2 s1^2 + 2 w (1-w) rho s1 s2
```

At w = 1/2 and s1 = s2 this reduces to the formula of section 1, sigma^2 (1 + rho) / 2. The minimising weight is

```
w* = (s1^2 - rho s1 s2) / (s1^2 + s2^2 - 2 rho s1 s2)
```

Inputs from `diversity_20261116_234858.csv`: s1 = 0.127183 (Lasso), s2 = 0.119007 (CatBoost), rho = 0.85637.

| Quantity | Value |
|---|---|
| Predicted RMSE of the equal-weight average | 0.11860 (pooled), which is 0.00041 below CatBoost pooled 0.11901 |
| Optimal weight | 0.728 on CatBoost, 0.272 on Lasso |
| Predicted RMSE at the optimal weight | 0.11763 (pooled), which is 0.00138 below CatBoost pooled |
| Break-even correlation (the equal-weight average ties CatBoost) | rho = (3 s2^2 - s1^2) / (2 s1 s2) = 0.869 |
| Measured, E068 average (fold mean) | 0.117344, which is 0.00073 below E066 0.118072 |
| Measured, E069 NNLS blend (fold mean) | 0.118022, which is 0.00005 below E066 |

Reading it honestly:

- The inputs are pooled OOF RMSEs and the measurements are means of five fold RMSEs. They are not the same quantity: the fold mean is lower (CatBoost 0.118072 against 0.119007, Lasso 0.12545 against 0.12718). So compare differences, not levels. The predicted gain of the average is -0.0004 and the measured gain is -0.0007: same sign and order of magnitude. The formula assumes a fixed correlation and ignores the weights being estimated, so a 0.0003 gap is not a contradiction.
- The average is close to break-even. The observed rho (0.856) is only 0.013 below 0.869, so the average beats CatBoost by a thin margin that would vanish at 0.869. This is the mechanism behind the shrinkage: the weaker Lasso (0.127 against 0.119) must pay for itself through decorrelation, and the better CatBoost gets, the less it can. With the Stage 1 inputs (CatBoost 0.1204, Lasso 0.1272, rho 0.860 from section 1, all rounded) the same formula predicts a gain of about -0.0010, against about -0.0004 now. The measured gains went from -0.0014 to -0.0007. The formula gets the direction and roughly the size of the shrinkage.
- The optimal weight is 0.73 on CatBoost, not 0.5, and would give a larger predicted gain (-0.0014 pooled). But that weight is computed from the same residuals it would be scored on, and a learned blend has to estimate it from inner folds. Measured, the NNLS blend (E069) did worse than the equal-weight average, by 0.0007. Averaging at 0.5 forgoes about 0.001 of the theoretical optimum and has no parameter to overfit.
- The same structure holds for more members: each new member must be accurate and decorrelated from the existing average. Boosters correlate 0.94 to 0.97 with CatBoost and the other linear models 0.97 to 0.997 with Lasso, so additions bring mostly shared errors plus their own bias.

### 10. Finalists on fresh folds (E076 to E083)

Rule (decision D17, audit F20): finalists are re-scored on folds disjoint from the selection partition. Selection used seed-42 five-fold. Finalists use RepeatedKFold 5 splits x 3 repeats, seed 7, giving 15 fold scores each. Each finalist has the same configuration as its source experiment, and only the splitter changes. Sources: `experiments/finalists/finalists_20261117_003230.csv` and registry records E076 to E083 (the `notes` field names the source).

#### 10.1 Selection score, fresh score, optimism

Optimism is fresh score minus selection score (positive means the selection-fold number was too good).

| Finalist | Source | Model | Selection (seed 42) | Fresh (seed 7, 15 folds) | Fold std | Optimism |
|---|---|---|---|---|---|---|
| E077 | E068 | average Lasso + CatBoost | 0.11734 | 0.12066 | 0.01433 | 0.00331 |
| E081 | E061 | average Lasso + Stage 1 CatBoost | 0.11807 | 0.12147 | 0.01488 | 0.00340 |
| E078 | E072 | stack Lasso + CatBoost | 0.11748 | 0.12164 | 0.01398 | 0.00416 |
| E076 | E066 | CatBoost Stage 2 (75 trials) | 0.11807 | 0.12166 | 0.01305 | 0.00359 |
| E079 | E069 | NNLS blend Lasso + CatBoost | 0.11802 | 0.12240 | 0.01393 | 0.00438 |
| E080 | E060 | CatBoost Stage 1 (25 trials) | 0.11946 | 0.12325 | 0.01286 | 0.00379 |
| E083 | E067 | LightGBM Stage 2 | 0.12173 | 0.12524 | 0.01320 | 0.00350 |
| E082 | E056 | Lasso grid | 0.12545 | 0.12816 | 0.01865 | 0.00271 |

Every finalist scores worse on fresh folds. The mean optimism is 0.0036 (range 0.0027 to 0.0044). This is the size of the selection effect: a configuration chosen as the best of many tries on the same folds is biased low. It exceeds the project's 0.003 rule for calling one model better than another, which is why finalists are never compared on selection scores. The two learned-weight finalists (stack E078 0.0042, blend E079 0.0044) have the largest optimism, which fits a meta-learner adding a little fit to the selection folds, but with differences of 0.001 this is a suggestion, not a finding.

#### 10.2 Paired differences against single CatBoost E066

Fold i of every finalist is paired with fold i of E076, since all eight used the same seed-7 splitter. Difference is finalist minus E076 (negative is better). SE is the sd of the 15 differences divided by sqrt(15). "Folds won" counts folds where the finalist beat E076. The repeat columns are the mean difference over the five folds of each repeat.

| Finalist | Model | Mean diff | SD of diffs | SE (naive) | Folds won | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---|---|---|---|---|---|---|---|
| E077 | average E068 | -0.00101 | 0.00683 | 0.00176 | 9/15 | -0.00074 | -0.00102 | -0.00126 |
| E081 | average E061 (Stage 1 CatBoost) | -0.00020 | 0.00815 | 0.00210 | 9/15 | +0.00010 | -0.00018 | -0.00051 |
| E078 | stack pair E072 | -0.00002 | 0.00621 | 0.00160 | 10/15 | -0.00016 | +0.00060 | -0.00050 |
| E079 | blend pair E069 | +0.00074 | 0.00584 | 0.00151 | 9/15 | +0.00070 | +0.00111 | +0.00041 |
| E080 | CatBoost Stage 1 E060 | +0.00158 | 0.00250 | 0.00064 | 5/15 | +0.00165 | +0.00154 | +0.00157 |
| E083 | LightGBM E067 | +0.00357 | 0.00363 | 0.00094 | 4/15 | +0.00444 | +0.00275 | +0.00353 |
| E082 | Lasso E056 | +0.00650 | 0.01473 | 0.00380 | 6/15 | +0.00673 | +0.00670 | +0.00607 |

Reading:

- **The average E077 beats E066 in all three repeats** (-0.0007, -0.0010, -0.0013) and in 9 of 15 folds. The mean difference is -0.0010 with a naive SE of 0.0018, 0.57 SE from zero. It is not distinguishable from zero.
- **Stage 2 CatBoost beats Stage 1 CatBoost on fresh folds.** E080 minus E076 is +0.0016, SE 0.0006 (2.5 naive SEs), and positive in all three repeats (+0.0015 to +0.0017). E080 wins only 5 of 15 folds, so the gain is small per fold and consistent. This supports D21 on independent folds.
- **The stack pair and the Stage 1 average are tied with E066** (-0.00002 and -0.0002). The learned blend E079 is slightly worse (+0.0007).
- **LightGBM and Lasso are clearly worse than CatBoost** (+0.0036 and +0.0065).
- The finalists CSV column `paired_diff_vs_best` is measured against E077 (the best fresh score), not against E066, so its values differ from this table (for example +0.00101 for E076, which is E076 minus E077). Use the table here for comparisons against E066.

#### 10.3 The 15 fold scores are not independent

The SE above treats the 15 differences as independent, and they are not. Each repeat splits the same 1,241 rows into five folds, so every row is in a test fold once per repeat, and training sets (80% of the rows each) overlap heavily across folds and repeats. Three repeats carry far less than three times the information of one 5-fold CV. The naive SE, sd/sqrt(15), understates the true standard error.

Nadeau and Bengio (2003) correct for this. For J = k x r = 15 resampled differences with sample variance s^2, the corrected variance of the mean difference is

```
Var_corrected = (1/J + n_test/n_train) * s^2
```

The second term does not shrink with more repeats. It stands for the dependence between folds, so more repeats can only reduce the first term.

For E077 against E076: s = 0.00683, J = 15, n_test = 248, n_train = 993 (5-fold on 1,241 rows). The factor is 1/15 + 248/993 = 0.0667 + 0.2497 = 0.3164, so

```
corrected SE = 0.00683 * sqrt(0.3164) = 0.00384
```

This is 2.2 times the naive 0.00176. The mean difference of -0.00101 is then 0.26 corrected SEs from zero (0.57 naive). The direction is the same and the evidence is weaker: the average-versus-CatBoost difference is far inside the noise of this protocol. The same correction on E080 minus E076 (s = 0.00250) gives a corrected SE of 0.00141 and a ratio of 1.1, so even the Stage 2 CatBoost gain is not conclusive on its own. The correction is an approximation derived for random resamples, so read the corrected SE as the more cautious of the two, not as exact.

### 11. What should be frozen? A balanced view

Candidates: (A) CatBoost E066 alone, (B) the equal-weight average of Lasso E056 and CatBoost E066 (the E068 configuration). The learned blend and the stack are not competitive: they cost far more fitting time on the fresh folds (E072 489.6 s and E069 529.0 s, against 97.4 s for the average and 84.3 s for CatBoost) and were not better.

For single CatBoost:

- It is the simplest candidate: one model, one set of hyper-parameters, one fit.
- No ensemble beats it by more than one SE on the selection folds, and none reaches the project's 0.003 rule on fresh folds (best, E077, -0.0010). With the fold-dependence correction the best difference is 0.26 SE.
- One model is easier to explain and audit (importances, partial dependence, SHAP).
- Trees do not extrapolate. On test-only extreme houses such as Id 2550, whose features lie outside the training range, a tree prediction stays within the range of the training targets, while a linear member can follow a slope beyond it. An equal-weight average passes half of the linear member's extrapolation through.
- The measured advantage of the average (0.0010) is inside the noise, so adding a second model would be paid for by complexity without evidence of a return.

For the equal-weight average:

- It was never worse than CatBoost in any of the three repeats (-0.0007, -0.0010, -0.0013) and won 9 of 15 folds.
- It has no learned weights, so there is no F18 risk and nothing to overfit. Its fit time is 97.4 s against 84.3 s.
- It hedges model-class risk. If the test distribution differs from the development one, two model classes with residual correlation 0.856 are less likely to fail together than one.
- By the variance formula in 9.4 the expected gain is positive as long as the residual correlation stays below 0.869. It holds on the measured value, with a thin margin.
- Predicted gain (-0.0004) and measured gains (-0.0007 on selection folds, -0.0010 on fresh folds) agree in sign.

Points that cut both ways: the expected gain of the average is about 0.0005 to 0.001 RMSLE, which is small against the fold-to-fold spread (std 0.013 to 0.015). It is unlikely to be visible on the leaderboard either way. The risk of the average is the linear member on out-of-range rows. The risk of CatBoost alone is giving up a small, consistent hedge.

**Director's ruling (D22, 2026-11-17).** Single CatBoost E066 is frozen. The equal-weight average has the best point estimate on both sets of folds, but its advantage (0.0010) is a third of a standard error and fails the rule the auditor wrote before reading the finalist records (at least 0.0010 better, beyond one standard error, at least 10 of 15 folds). Ties go to the simpler model. This is a parsimony ruling, not evidence that the average is worse; part 4 gives the full account.

### 12. Self-check questions

1. For two models with equal error variance, how does the variance of their average depend on error correlation? *Variance falls to sigma^2 (1 + rho) / 2. At rho = 1 there is no gain, at rho = 0 it halves.*
2. Why is prediction correlation a poor diversity measure here? *All models predict the same price scale, so it is above 0.9 for almost any pair. Residual correlation removes the shared signal and shows the shared mistakes.*
3. Why constrain blend weights to be non-negative and sum to one? *Negative weights exploit noise, and a free sum adds a scale parameter. The constraints act as regularisation.*
4. What is the F18 risk? *Choosing weights from OOF scores on the same folds later used to report the blend makes the score optimistic. We learn weights inside `fit` or use equal weights.*
5. Why use Ridge and `passthrough=False` in the meta-learner? *Base predictions are strongly correlated, so a low-capacity regularised model is stable, and without raw features the meta-model cannot relearn the whole problem.*
6. Walk through one outer fold of nested stacking. *Inner K-fold on outer-train rows gives an OOF matrix. Fit the meta-learner on it. Refit the bases on all outer-train rows. Predict the outer validation fold once through bases then meta.*
7. Why must the meta-learner train on OOF and not in-sample base predictions? *In-sample predictions are too accurate, so it over-trusts flexible models, then fails on unseen rows where bases are less accurate.*
8. What does `OptimisticScore` prevent, and what does it not? *It stops the optimistic number from flowing through `run_and_register`, and a 0.0 value from triggering a real CV. It cannot stop someone building a registry record by hand.*
9. What was the F16 blind spot and how is it closed? *A lone memoriser returns a constant on unseen rows, so a leaky and an honest stack score alike. Adding a Ridge base gives the leak something to damage: honest stays near Ridge, leaky is much worse.*
10. How many fits does a nested stack of six bases need per outer fold and in total? *6 x 6 = 36 per outer fold and 180 over five folds, versus 30 for plain CV.*
11. As CatBoost improved from 0.11946 to 0.11807, why did the gain from averaging with Lasso halve? *The average only helps while the residual correlation stays below a break-even that depends on the accuracy gap. Here it is 0.869 (observed 0.856). A more accurate CatBoost lowers the break-even and leaves less room for the weaker Lasso. Measured gain went from -0.0014 to -0.0007.*
12. With s1 = 0.1272, s2 = 0.1190 and rho = 0.856, what weight minimises the average's error, and why use 0.5 anyway? *w* = 0.73 on CatBoost, with a predicted pooled RMSE of 0.1176 against 0.1186 at 0.5. But the optimum is estimated from the same residuals it is scored on, and the learned blend E069 measured worse than the plain average. Equal weights give up about 0.001 and have no parameter to overfit.*
13. Why is sd/sqrt(15) too small for a RepeatedKFold 5x3 difference, and what does Nadeau-Bengio change? *Repeats reuse the same rows and training sets overlap, so the 15 folds are dependent. The corrected variance is (1/15 + n_test/n_train) s^2, whose second term does not shrink with repeats. For E077 against E076 the SE rises from 0.0018 to 0.0038.*
14. Why do finalists have to be re-scored on fresh folds, and how large was the optimism? *Selection picks the best of many tries on the same folds, so the score is biased low. Fresh-fold scores were worse for all eight finalists, by 0.0036 on average (range 0.0027 to 0.0044).*


---

## Study guide, part 4: from finalists to a frozen model

Parts 2 and 3 produced candidates. This part explains how one of them became the final model, why the choice was made the way it was, what the holdout did and did not tell us, and how the submission file was produced. The records are ruling D22 in `docs/EXPERIMENT_HISTORY.md`, section 9 and 10 of `docs/VALIDATION_AUDIT.md`, registry ids E076 to E083, and `experiments/holdout_usage.log`.

### 1. Why the best score on the selection folds is not the answer

By the time the candidates existed, the five selection folds (seed 42) had been used to compare 136 linear grid points, 180 boosting trials and 12 ensembles. Each comparison picks the configuration that happens to suit those five particular splits. The picked score is therefore biased low: it contains the true quality of the configuration plus the luck that made it win. This is the winner's curse, and it grows with the number of things compared.

The cure is cheap: score the chosen configurations again on folds they were not chosen on. `experiments/finalists/run_finalists.py` rebuilds each candidate from the registry and runs `RepeatedKFold(n_splits=5, n_repeats=3, random_state=7)`. Seed 7 gives partitions that share no fold with the seed-42 partition (audit finding F20 was precisely that the first version reused seed 42, which made one of the three repeats identical to the selection folds).

The result, for the eight candidates:

| model | selection folds | fresh folds | optimism |
|---|---|---|---|
| average Lasso + CatBoost (E068) | 0.1173 | 0.1207 | +0.0033 |
| nested stack Lasso + CatBoost (E072) | 0.1175 | 0.1216 | +0.0042 |
| CatBoost, 75 trials (E066) | 0.1181 | 0.1217 | +0.0036 |
| NNLS blend Lasso + CatBoost (E069) | 0.1180 | 0.1224 | +0.0044 |
| CatBoost, 25 trials (E060) | 0.1195 | 0.1233 | +0.0038 |
| LightGBM, 75 trials (E067) | 0.1217 | 0.1252 | +0.0035 |
| Lasso (E056) | 0.1255 | 0.1282 | +0.0027 |

Every candidate got worse, by 0.0027 to 0.0044. Two things are mixed in that shift. One is the fold draw: the seed-7 partitions are simply a little harder for all models (the least-tuned model, Lasso, also moved by 0.0027). The other is the winner's curse, which should be larger for configurations picked from more alternatives (the blend and the stack, picked among ensembles built on bases that were themselves picked, moved most). You cannot separate the two exactly, and you do not need to: the fresh-fold number is the one to report.

### 2. Comparing two models properly: paired differences

Fold scores vary a lot (for CatBoost from 0.103 to 0.150 across the 15 fresh folds) because some folds contain hard houses. Comparing two means with that spread would hide any real difference. But both models are scored on the same folds, and a hard fold is hard for both. So the comparison is made on the per-fold difference d_i = score_A,i − score_B,i, which removes the fold difficulty.

For the average (E068) against single CatBoost (E066): mean difference −0.0010, standard deviation of the 15 differences 0.0068.

How precise is that mean? Three answers, from most to least optimistic:

- Naive: sd / sqrt(15) = 0.0018. This treats the 15 folds as independent measurements. They are not: the three repeats cut the same 1,241 houses three ways, and within a repeat the training sets overlap by 75 %.
- The auditor's choice (finding F30): sd / sqrt(5) = 0.0031, on the argument that the three repeats agree with each other almost perfectly and so contribute little new information beyond one 5-fold run.
- Nadeau and Bengio's corrected resampled variance: multiply the variance by (1/15 + n_test/n_train) = (1/15 + 248/993) = 0.316 instead of 1/15, giving 0.0068 × sqrt(0.316) = 0.0038.

Whichever you take, −0.0010 is between a quarter and a half of one standard error. The data cannot tell these two procedures apart. It also cannot exclude that the average is really better by 0.002 or 0.003; "not significant" is not "equal".

Contrast with CatBoost 75 trials against CatBoost 25 trials: difference 0.0016 with a standard deviation of only 0.0025, because the two models are almost the same model and their fold scores move together. That is 1.1 to 2.5 standard errors depending on the formula, and the three repeat means are 0.0017, 0.0015, 0.0016. Small, but consistent.

### 3. The decision rule was written before the data was seen

If you look at a table and then invent the rule that justifies your favourite row, the rule is worthless. The auditor therefore wrote the rule into `docs/VALIDATION_AUDIT.md` before reading the finalist records:

> Single CatBoost stays unless an ensemble is at least 0.0010 better on the fresh folds, and the difference exceeds one standard error (sd/sqrt(5)), and it wins at least 10 of the 15 folds with all three repeat means negative.

The reasoning behind the asymmetry (the simple model wins ties): an ensemble costs a second pipeline to maintain, explain and audit; its seed-42 advantage is inflated by having been the best of eight ensembles; and the project's standing rule since audit round 4 is that differences under 0.003 are not wins.

The average met two conditions (−0.0010; all three repeats negative) and failed two (one third of a standard error; 9 folds of 15). Ruling D22 froze single CatBoost E066.

Be precise when you explain this: it is a parsimony decision. The honest sentence is "the average had the best point estimate on both sets of folds, the difference was far inside the noise, so I kept the simpler model", not "the ensemble did not help".

A second argument, specific to this data, is about the test set. Test house Id 2550 has 5,095 square feet of living area, larger than all but one training house. A linear model multiplies area by a coefficient and so extrapolates; the two training houses of that kind (Ids 524 and 1299) sold far below what a linear trend predicts. A tree model predicts for Id 2550 what it learned for the largest houses it has seen. Neither is guaranteed to be right, but the tree's failure mode is bounded.

### 4. What "frozen" means

D22 records the registry id (E066), the configuration hash (`be58905cb40a`), the builder (`houseprices.models:build_catboost`), the six hyper-parameters at full precision, the feature set, the row filter and the target. The freeze was committed to Git (`23f56ac`) before anything else happened, so the history proves the order of events. From that commit on, nothing about the procedure may change in response to what the holdout says. If the holdout had come back terrible, the correct response would have been to look for a bug and report the bad number, not to retune.

### 5. The single holdout evaluation

Before the evaluation the auditor also wrote down what result would be consistent with the cross-validated estimate: centre 0.121, consistent from 0.100 to 0.145, a bad draw up to 0.150, and outside 0.095 to 0.150 a sign of a data or pipeline fault.

Then, in order:

1. `python -m houseprices.train --experiment E066` fitted the frozen pipeline on the 1,241 development rows and saved `outputs/models/E066_development.joblib` with a JSON sidecar (rows, hash, commit, library versions).
2. `experiments/finalists/score_holdout.py` checked the sidecar (right experiment, development rows only, no override), then called `validation.score_holdout_once` once.

`score_holdout_once` is the guard described in part 2a: it refuses a frame whose Ids are not exactly the 219 registered holdout Ids, refuses a second call, and appends the event to `experiments/holdout_usage.log`.

Result: **RMSE 0.10645** on 219 houses. Mean residual +0.004 (no bias), mean absolute residual 0.076, three houses with an absolute error above 0.3, median absolute error in dollars 5.3 %.

### 6. Why the holdout score is better than the CV estimate, and why that changes nothing

0.1065 against 0.1217 looks like good news. Treat it with the same suspicion you would apply to bad news.

- Sampling error. An RMSE over 219 houses has a standard deviation of roughly 0.01 to 0.014 here, because a handful of large errors dominate a squared-error average. A gap of 0.015 is about one standard deviation.
- The hard houses are elsewhere. The four houses above 4,000 square feet (Ids 524, 692, 1183, 1299) carry a large part of the development-set squared error (part 5 quantifies it), and all four are development rows. The stratification of the holdout was by price decile, not by "difficulty". Of the 15 fresh folds, the five that contain none of them average 0.114 (0.103 to 0.125) and the ten that contain at least one average 0.126 (up to 0.150); the holdout, with none, sits in the first group.
- The model fitted for the holdout used 1,241 rows, CV models about 993. More data helps a little.

So the holdout says: the cross-validated estimate was not inflated by leakage or by overfitting the folds; nothing is broken. It does not say the model's expected error is 0.106. The number to quote for expected performance on new houses remains about 0.12, and you should expect Kaggle's test set, which contains at least one giant house, to behave more like the CV estimate than like the holdout.

The project had predicted (hypothesis H4) that the holdout would land within 0.01 of the fresh-fold mean. It landed 0.015 away. The prediction was wrong and the experiment history says so.

### 7. The final fit and the submission file

Once a holdout has been scored it has no further use as a test, and its 219 rows are worth more as training data. `train --rows all` refits the frozen configuration on all 1,460 rows. The code refuses this before the holdout has been scored (audit finding F24): a model that has seen the holdout must never be scored on it.

`python -m houseprices.predict --model outputs/models/E066_all.joblib` then applies the same cleaning and feature code to `test.csv`, predicts on the log scale, converts with `expm1`, checks that every price is finite and positive, aligns the rows to the Ids of `sample_submission.csv` and writes `outputs/submissions/E066_all.csv` (1,459 rows, prices from 41,901 to 524,842, median 156,611).

Things to notice:

- There is no cross-validated score for this last model and there cannot be one. Its expected error is inferred from the procedure's CV score. That is why the procedure, not a fitted object, is what gets frozen and evaluated.
- The predicted prices stop at 525 thousand while training prices reach 755 thousand. A model that minimises squared error on the log scale shrinks its predictions toward the centre; part 5 shows the same pattern in the residuals by price decile.
- The file was not submitted. Kaggle's public leaderboard is one more score on one more sample; using it to adjust the model would turn it into a tuning set.

### Self-check questions

1. Why did every finalist score worse on the seed-7 folds, including Lasso, which was chosen from a small grid?
2. The average beat CatBoost by 0.0010 on fresh folds and by 0.0007 on the selection folds. Why was it not chosen?
3. Why is sd/sqrt(15) too small a standard error for a 5x3 repeated cross-validation?
4. What would you have done if the holdout RMSE had been 0.16?
5. The holdout RMSE (0.1065) is lower than the CV estimate (0.1217). Which do you report as expected performance, and why?
6. Why is `train --rows all` refused before the holdout is scored?
7. Why is the decision rule's timing (written before the table was read) part of the evidence?

### Answers

1. Two effects: the seed-7 partitions are a slightly harder draw for all models, and scores chosen as the best of many on seed 42 include favourable luck that does not repeat. Lasso's shift (0.0027) is mostly the first effect.
2. The difference is a quarter to a half of its standard error; the pre-registered rule required more than one standard error and 10 of 15 folds; ties go to the simpler model. It is a parsimony choice and the report says the average may be slightly better.
3. The 15 fold scores share the same 1,241 rows; repeats re-cut the same data and training sets overlap heavily, so the scores are positively correlated and the effective number of independent measurements is much smaller than 15.
4. Look for a fault (wrong rows, a preprocessing difference between fit and predict, a data problem in the holdout), report the number as it is, and not retune. If anything were changed because of it, the holdout would no longer be an unbiased estimate and the report would have to say so.
5. The CV estimate, about 0.12. The holdout is a single sample of 219 houses that happens to contain none of the four hardest houses; it is a consistency check that came out on the favourable side of its expected range.
6. A model fitted on all rows has seen the holdout rows; scoring it there would measure memorisation. The guard makes the mistake impossible instead of merely discouraged.
7. A rule chosen after seeing the results can always be made to prefer any candidate. Committing the rule first (it is in the audit file, written before the finalist records were read) makes the choice a test, not a rationalisation.


---

## Study guide, part 5: what the final model relies on and where it fails

Source: `experiments/interpretability/run_interpretability.py --experiment E066`. All numbers come from the tables in `outputs/interpretability/` (prefix `E066_`) unless stated. Errors are on the log1p(SalePrice) scale; residual = actual minus predicted, so a negative residual is an over-prediction. The out-of-fold (OOF) predictions use the selection folds (KFold 5, shuffle, seed 42) on the 1,241 development rows and reproduce the registered fold scores (CV mean 0.11807). The pooled OOF RMSE is 0.1190, slightly different from the mean of the fold RMSEs because folds differ in size. These are selection-fold numbers, so they are somewhat optimistic (about +0.002 to +0.003); the fresh-fold estimate for E066 is 0.1217.

### 1. What the model relies on

Two methods, with different meanings.

**Permutation importance** (`E066_permutation_importance.csv`, figure `outputs/figures/E066_permutation_top25.png`). For each fold the model is fit on the training part, then one raw input column at a time is shuffled in the validation part (3 repeats) and the rise in RMSE is recorded. This measures how much real, out-of-sample predictive information the column carries. The ranking:

| column | RMSE increase | std across folds |
|---|---|---|
| OverallQual_x_TotalSF | 0.0637 | 0.0107 |
| OverallQual_x_GrLivArea | 0.0214 | 0.0075 |
| TotalSF | 0.0166 | 0.0044 |
| OverallCond | 0.0079 | 0.0023 |
| MSZoning | 0.0058 | 0.0026 |
| OverallQual | 0.0057 | 0.0029 |
| LotArea | 0.0056 | 0.0015 |
| Neighborhood | 0.0056 | 0.0017 |
| TotalBath | 0.0055 | 0.0014 |

Of the 99 raw columns, only 32 raise the error by more than 0.001 and 16 have zero or negative importance. The top three alone account for about 53 percent of the summed positive importance (0.102 of 0.191).

**SHAP** (`E066_shap_mean_abs_by_raw_column.csv`, `outputs/figures/E066_shap_top25_bar.png`, `E066_shap_top25_raw_bar.png`, `E066_shap_beeswarm.png`, `E066_shap_dependence_1.png`, `E066_shap_dependence_2.png`). SHAP describes how the fitted model (fit on all development rows) distributes credit for its predictions; it is in-sample and says nothing about generalisation. The one-hot levels are summed back to their raw column. The top raw columns by summed mean |SHAP| are OverallQual_x_TotalSF 0.0714, OverallQual_x_GrLivArea 0.0324, TotalSF 0.0255, OverallCond 0.0199, Neighborhood 0.0195, TotalBath 0.0186, LotArea 0.0146, OverallQual 0.0146 and MSZoning 0.0129. The top five hold 28 percent of the total.

**Agreement and the correlated-features caveat.** Both methods put the same three size-and-quality columns first. They disagree only in the middle (for example Neighborhood is 8th by permutation and 5th by SHAP, because its many one-hot levels are summed in SHAP but shuffled as a unit in the permutation test). The caveat matters: OverallQual_x_TotalSF, OverallQual_x_GrLivArea, TotalSF, GrLivArea and OverallQual are strongly related (GrLivArea is a component of TotalSF, and the interactions contain both factors). Shuffling one leaves the others intact, so each looks less important than the group; SHAP splits credit among them in a way that depends on the trees. The honest reading is "size times quality is what the model uses", not that the interaction column is individually 4 times as valuable as TotalSF.

### 2. Where it fails

**By price** (`E066_residuals_by_price_decile.csv`, `outputs/figures/E066_residual_by_decile.png`, `E066_pred_vs_actual.png`, `E066_residual_vs_predicted.png`). The error is U-shaped: RMSE 0.215 in the cheapest decile (houses at or below 106,500) and 0.153 in the most expensive (278,000 and up), against 0.076 to 0.106 in deciles 2 to 9. The mean residuals show regression toward the mean: -0.097 for the cheapest decile (over-predicted) and +0.072 for the most expensive (under-predicted).

**By living area** (`E066_residuals_by_grlivarea.csv`). RMSE is 0.093 for 1,500 to 1,999 sq ft (393 houses), 0.134 below 1,000, 0.125 for 2,000 to 2,499, 0.201 for 3,000 to 3,999 (12 houses, mean residual +0.101) and 0.611 for 4,000 and above (4 houses).

**By neighbourhood** (`E066_residuals_by_neighborhood_top_bottom5.csv`). Worst RMSE: IDOTRR 0.239 (n=29), Edwards 0.165 (n=86), NoRidge 0.162 (n=34), StoneBr 0.157 (n=21), OldTown 0.156 (n=95). Best: CollgCr 0.069 (n=130), BrDale 0.066, Gilbert 0.063; NPkVill (n=7) and Blueste (n=2) are too small to read.

**The worst rows** (`E066_worst15_rows.csv`). The largest residual is Id 633 (-0.748, NWAmes, sold as a Family sale), then Ids 31, 463, 524 (-0.669), 1325, 1183 (+0.636) and 969. Four of the 15 are the houses above 4,000 sq ft (Ids 524, 692, 1183, 1299). Most of the others are cheap, old houses in IDOTRR, OldTown and Sawyer, or Partial and Abnormal sales, where the price probably reflects circumstances (condition of the sale, buyer) that the features capture only partly (my interpretation).

**How concentrated.** The worst 5 rows carry 13.6 percent of the total squared error and the worst 15 carry 31.1 percent, while only 6.6 percent of houses have an absolute residual above 0.2. Removing the four houses with GrLivArea above 4,000 lowers the pooled OOF RMSE from 0.1190 to 0.1140 (1,237 rows). Those four are predicted at 360,787 (actual 184,750, Id 524), 292,371 (160,000, Id 1299), 444,991 (755,000, Id 692) and 394,416 (745,000, Id 1183): the two Partial sales are over-predicted by a factor of about 1.8 to 1.95, the two NoRidge houses under-predicted by a factor of about 1.7 to 1.9.

### 3. Implications for the test set

Test Id 2550 is a 5,095 sq ft Edwards partial sale, the same profile as training Ids 524 and 1299, which sold for 184,750 and 160,000 and were over-predicted by 0.67 and 0.60. The submission predicts 268,438 for it. Two points follow. First, a tree model cannot extrapolate: every prediction is a sum of leaf values learned from training rows, and with only four training houses above 4,000 sq ft the model has almost no basis for a 5,095 sq ft house; 268,438 is a blend of the neighbours in the feature space, not an estimate of what such a house is worth. Second, whether the actual price resembles Ids 524 and 1299 (low, partial-sale effect) or Ids 692 and 1183 (very high) is exactly the question the training data cannot settle: the two pairs have opposite residual signs. One house cannot move the leaderboard score much, but expect this prediction to be wrong by a large factor in either direction. It should be treated as a known uncertainty, not tuned by hand.

### 4. Holdout residuals compared with the OOF pattern

The single holdout evaluation (219 houses, RMSE 0.10645) saved `experiments/finalists/holdout_predictions_E066.csv`; I only read the file. Splitting the holdout into terciles of actual price gives mean residual and RMSE of -0.023 and 0.114 (low tercile, 76 houses), +0.001 and 0.100 (middle, 71) and +0.035 and 0.104 (high, 72). The same terciles of the development OOF predictions give -0.035 and 0.144, +0.004 and 0.096, +0.030 and 0.113. The sign pattern is the same (cheap houses over-predicted, expensive under-predicted, nothing in the middle) with a smaller magnitude on the holdout. The holdout RMSE of 0.1065 is below the CV estimate of 0.1217, which fits a holdout with milder extreme errors (I did not inspect its areas) rather than a better model: its largest residuals are 0.41 and 0.40 (Ids 711 and 875), against 0.75 in development.

### 5. Limits of this analysis

- OOF residuals are from the selection folds and the model was selected on them; absolute levels are optimistic by 0.002 to 0.003.
- Permutation importance breaks the correlation structure when it shuffles a column, so it evaluates the model on unrealistic rows; it also shares credit among related columns.
- SHAP is in-sample and conditional on the model, and a sum over one-hot levels is a convention.
- Several groups are tiny (4 houses above 4,000 sq ft, 2 in Blueste, 7 in NPkVill); their RMSEs are anecdotes.
- Nothing here is causal. A column being important means the model uses it, not that changing it would change a price.


---

## Study guide, part 6: the Python implementation

The earlier parts quote the code where they need it. This part gives the map: which module does what, how a single experiment flows through the package, and the handful of design choices that make the results reproducible. The package is about 2,600 lines in `src/houseprices/`, with roughly 215 tests in `tests/`.

### 1. The data flow of one experiment

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

### 2. The modules

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

### 3. Why everything is a scikit-learn Pipeline

`cross_validate` does, for each fold, `est = clone(estimator); est.fit(X_train, y_train); est.predict(X_valid)`. Because the estimator is a `Pipeline` whose first step is the preprocessor, `fit` on the training rows fits the imputers, the skew selector, the encoder and the model in one call, and `predict` on the validation rows only applies them. There is no code path in which a transformer could be fitted on validation rows, short of writing one on purpose. Leakage safety is a property of the structure, not of discipline.

The same idea carries the ensembles. A `BlendingRegressor` is an estimator whose `fit` runs an inner 5-fold cross-validation on the rows it is given, learns the weights from those inner out-of-fold predictions, and refits the base models on all its rows. When the outer `cross_validate` calls `fit` on an outer-training fold, the weights are learned without the outer-validation rows. Nesting is automatic.

`clone` matters: it creates an unfitted copy with the same parameters, so no state survives from one fold to the next.

### 4. Builders, keyword arguments and the configuration hash

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

### 5. The experiment runners

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

### 6. Guards written as code

Several rules of the methodology are enforced by the program and tested, so that breaking them requires deliberate effort:

- `cross_validate` raises if any prediction is not finite or if a training-row filter empties a fold (audit F2, F3). A silent NaN score would otherwise be stored and sorted last.
- Training-row filters receive only the training part of a fold; a test asserts that validation indices are identical under every filter.
- `score_holdout_once` checks the Id set, refuses a second use, and logs.
- `train --rows all` raises until the holdout log is non-empty.
- `run_finalists.py` refuses seed 42.
- `predict` refuses a submission whose Ids differ from `sample_submission.csv` or whose prices are not finite and positive.

The tests in `tests/test_audit_round2.py` are the auditor's probes: each audit finding became a test that failed before the fix and passes after it.

### 7. Commands you should be able to type

```powershell
$py = "C:\Users\Enrique Baena\anaconda3\envs\houseprices\python.exe"
& $py -m pytest tests                                               # about one minute
& $py -m houseprices.evaluate --experiment E066                     # reproduce the frozen model's selection-fold score
& $py -m houseprices.evaluate --experiment E076                     # reproduce its fresh-fold score (15 folds)
& $py -m houseprices.train --experiment E066 --rows all             # refit the final model
& $py -m houseprices.predict --model outputs\models\E066_all.joblib # rewrite the submission file
```

### 8. Limits of the implementation

- The registry is a text file with no locking. Two processes appending at once can take the same id; the project avoided this by allowing one writer at a time (audit F27).
- Fitted models are saved with joblib, which is pickle: load only files you created (F28), and expect them to break across library versions. The sidecar JSON records the versions.
- The Optuna database is not in Git. The per-trial CSV files are, and the registered experiments carry the winning arguments, so results are reproducible without it; the search itself is not exactly re-creatable because the sampler was re-seeded when Stage 2 resumed (F31).
- Thread count affects the speed of the boosters, and in principle the last digits of their results. Registered scores reproduced to four decimals on the same machine; a different machine may differ slightly.

### Self-check questions

1. Why may `TotalSF` be computed before cross-validation while the LotFrontage median may not?
2. What exactly does `python -m houseprices.evaluate --experiment E066` prove, and what does `train` not prove?
3. Why are ensemble members passed as "specs" instead of estimator objects?
4. How does the structure of `Pipeline` plus `clone` prevent a scaler from seeing validation rows?

### Answers

1. `TotalSF` for a house uses only that house's columns. A neighbourhood median uses other houses; computed on all rows it would include validation houses in a statistic used to fill training and validation rows alike.
2. That the stored configuration, rebuilt from the registry and scored on folds rebuilt from the stored seed, gives the stored fold scores. `train` only produces a fitted object; it says nothing about how well the procedure generalises.
3. Registry arguments must be JSON so that they can be hashed, stored and rebuilt; a spec (builder name plus JSON arguments) can be, a Python object cannot.
4. `cross_validate` clones the whole pipeline for each fold and calls `fit` with training rows only; the scaler is a step of that pipeline, so its statistics come from those rows. On validation rows only `transform` is called.


---

## Study guide, part 7: defending the project

This part is the revision sheet. It lists every binding decision with the one-sentence reason you should be able to give, what was tried and rejected, the limitations you should volunteer before an interviewer finds them, and the questions most likely to be asked.

### 1. The decisions, with the sentence that defends each

| # | Decision | The sentence |
|---|---|---|
| D1 | Target `log1p(SalePrice)`, metric RMSE on that scale | It is the competition metric, and errors in log price are relative errors, which is how house prices are judged. |
| D2 | 5-fold shuffled CV (seed 42) for selection; 5x3 repeated CV for finalists | With 1,241 rows, 5 folds keep about 993 rows for training; repeats measure how much a score depends on the split. |
| D3, D15 | Outlier filters act on training folds only; final ruling: keep all rows | Deleting hard houses from validation makes the score better without making the model better; measured honestly, removal made every model worse. |
| D4 | 15 % holdout, stratified by price decile, single use | An estimate that no decision has touched; one use, because every reuse turns it into a tuning set. |
| D5, D6 | Rule B04 as a condition, rule C01 reconstructed and labelled as such | A rule tied to a row Id does not generalise; a lost cell is documented as reconstructed, not claimed as recovered. |
| D7 | pandas pinned to 2.3.3 | The original notebook's behaviour depends on the string dtype that pandas 3 changes. |
| D8 | CV raises on non-finite predictions and empty folds | A NaN score would be stored silently and sorted last. |
| D9 | `Id` travels with the features and is dropped inside every builder | Filters and the submission need it; no model may see it. |
| D11 | Feature set "all" (base + areas + baths + ages + quality interactions + ordinal quality) | Only the quality interactions clearly helped; pruning the others at default hyper-parameters would be fitting to noise. |
| D12 | Skew correction for linear models only | Worth about 0.009 for them; trees are invariant to monotone transforms. |
| D13 | LotFrontage: neighbourhood median, fitted in the fold | No alternative differed beyond noise; keep the documented default. |
| D14 | One-standard-error verdicts on 5 folds are provisional | Differences near 0.001 are noise at 5 folds. |
| D16 | Lasso and ElasticNet are the linear finalists | The L1 models beat Ridge by 0.005 in every fold. |
| D17 | Finalists are scored with seed 7 | Seed 42 folds were used for choosing; reusing them repeats the optimism. |
| D18, D20 | Test-only anomalies documented; year clipper clips the upper bound only | Problems invisible to CV must be reported; clipping the lower bound would have altered two real 1890s garages. |
| D19, D21 | Staged tuning: 25 trials each, then 75 for CatBoost and LightGBM, then stop | Spend compute where the first stage says it may pay; stop when the curve is flat and the gain is below the noise band. |
| D22 | Freeze single CatBoost E066 | The best ensemble's advantage was a third of a standard error; ties go to the simpler model, by a rule written before the comparison was read. |

### 2. What was tried and rejected

| Idea | Evidence | Why it matters |
|---|---|---|
| Drop the two (or four) large-area outliers | E049–E054: +0.004 to +0.009 worse for linear models, +0.001 to +0.005 for XGBoost | The usual gain in public notebooks comes from also deleting them from validation. |
| KNN or global-median LotFrontage | E043–E048: differences at most 0.002 | Imputation method for one column with 18 % missing values barely matters here. |
| Ordinal quality codes as a stand-alone improvement | ablations: within ±0.001 | One-hot already carries the information. |
| RobustScaler versus StandardScaler | linear grid: at most 0.0001 | Irrelevant once skewed columns are log-transformed. |
| Ridge as a finalist | 0.005 behind Lasso in every fold | With about 300 one-hot columns, sparsity beats shrinkage. |
| Native categorical handling in CatBoost | worse in all five paired trials by 0.0016–0.0028, at 15–24 times the cost; study stopped at 5 of 25 trials | A partial result, reported as partial. |
| More XGBoost tuning | Stage 1 flat from trial 4, 0.0036 behind | The Stage 2 rule excluded it. |
| NNLS blend weights, stacking with a Ridge meta-learner | never better than the equal-weight average on either set of folds | Learned weights add variance; with two members and correlated errors there is little to learn. |
| Ensembles of six bases or of three boosters | slightly worse than CatBoost alone | Members of the same family have residual correlations of 0.94–0.997; they add cost, not diversity. |
| Two-sided year clipping | would have changed two valid test rows | A "fix" must be checked against what else it touches. |

### 3. Limitations to state yourself

1. **The model choice at the top is within noise.** Single CatBoost and the Lasso + CatBoost average differ by 0.001 with a standard error of about 0.003. Another analyst could defensibly ship the average.
2. **Expected error is about 0.12, not the holdout's 0.106.** The holdout has 219 rows and none of the four largest houses.
3. **Selection optimism was measured, not removed.** The fresh-fold check corrects the reported number, but the same development rows were used throughout; only the holdout is truly untouched, and it is small.
4. **The search space had edges.** Both Stage 2 bests sit at the 2,000-tree upper bound; a wider search might find slightly better settings. The expected gain is below the noise band.
5. **A few houses dominate the error.** Large, atypically cheap or expensive houses are predicted badly, and the test set contains at least one (Id 2550). CV cannot tell which way that prediction errs.
6. **Regression toward the mean.** Cheap houses are over-predicted and expensive ones under-predicted (part 5).
7. **One city, 2006–2010.** Nothing here says how the model would behave on other markets or years; time was treated as a feature, not as a split.
8. **Tooling.** Registry without locking, pickled model artefacts, Optuna study not exactly re-creatable.
9. **No leaderboard score.** By choice, the file was not submitted, so there is no external confirmation.

### 4. Questions you should expect, with short answers

**What is your validation scheme and why should I trust the number?** Five-fold CV on development rows with all fitted preprocessing inside the fold; finalists re-scored on 15 new folds; one frozen model scored once on a locked holdout; every score reproducible by a command; an independent auditor's findings and fixes in the repository.

**What is data leakage and where could it have happened here?** Any path by which information about validation or test targets, or statistics computed with validation rows, reaches training. Candidates here: imputing with statistics from all rows, choosing skewed columns on all rows, removing outliers from validation folds, fitting ensemble weights on the predictions they are then scored on, tuning on the holdout. Each has a structural guard (part 2a, section 4).

**Why did you not use the ensemble? Everyone on Kaggle does.** On a leaderboard a gain of 0.001 moves you up many places and costs nothing. As an estimate of real performance it was not distinguishable from zero in my data, and I committed to a rule before looking. I report both numbers.

**Why CatBoost over XGBoost and LightGBM?** On this data it was better at every stage (0.1181 against 0.1217 and 0.1231 on the selection folds; 0.0036 better than LightGBM on fresh folds, about two standard errors). Its trial scores were also much less sensitive to hyper-parameters. Symmetric trees are a strong regulariser on small data. That is an empirical statement about 1,241 rows, not a general ranking.

**Why is your CV score worse than the 0.11 I see in public notebooks?** Mostly protocol: many public scores remove outliers from all rows, fit preprocessing on train and test together, and report the best of many attempts on the same folds or on the public leaderboard. On my selection folds the best procedure scored 0.117; measured honestly it is 0.121.

**What would you do with another week?** Target the errors rather than the average: a model or features for the large-house and low-price segments, a monotonic constraint on area, quantile or interval predictions; a wider search on the tree count; a time-aware validation split to check stability across sale years.

**What did the audit actually catch?** Real defects in code that passed casual review: out-of-fold predictions overwritten under repeated CV (F1), NaN predictions scored silently (F2), the Id column dropped before a filter needed it (F9), finalist folds identical to selection folds (F20), `train --rows all` possible before the holdout was scored (F24), and a standard error that treated correlated folds as independent (F30).

### 5. A mental model to keep

1. Split off a test set first and lock it.
2. Anything that learns from rows goes inside the fold; only row-wise arithmetic stays outside.
3. A score you used for choosing is biased; measure the chosen thing again on data that did not choose it.
4. Compare models with paired differences and an honest standard error; decide the rule before reading the result.
5. Prefer the simpler model when the data cannot tell them apart, and say that this is what you did.
6. Report what failed and what you cannot know.
