DRAFT, COMPLETE as of 2026-11-13 — sections done: Chapters 1-6 and the 10 self-check questions with answers; sections remaining: none (director review and merge only). Chapter 6 updated for D20 (YearRangeClipper).

# Study guide, part 1: the data, the cleaning and the features

Draft by Agent A (feature-researcher) for the director to merge into `docs/STUDY_GUIDE.md`. Audience: a third-year data-science student who must be able to defend every decision in an interview. Numbers come from `docs/AUDIT_2026-11-11.md`, `docs/EXPERIMENT_HISTORY.md` and the experiment registry (experiment ids are written E0xx). Cross-validated errors are RMSE on log1p(SalePrice), five shuffled folds, seed 42, lower is better.

---

## Chapter 1. The Ames data and the competition metric

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

## Chapter 2. The seventeen cleaning rules

### 2.1 The principle: semantic fills, not statistical fills

In this data a missing value usually does not mean "unknown measurement". It often means the feature does not exist. `PoolQC` is blank for 1,453 of 1,460 houses because those houses have no pool, and `data_description.txt` says so. Replacing that blank with the most common quality grade would invent pools. The cleaning rules therefore fill by **meaning**: "NoPool", "NoBasement", "NoGarage", and so on. A semantic fill has two properties we rely on.

- It uses only the same row (and the data dictionary). Nothing is estimated from other houses, and `SalePrice` is never read. So the rules cannot leak information between training and validation data and may safely run once, before any cross-validation split.
- It never overwrites a recorded value. Rules only fill a missing entry or add a flag column. Contradictions (for example a pool area above zero but no pool quality) are flagged and marked "Unknown", not "fixed".

A **statistical** fill, such as replacing a missing `LotFrontage` with a median, is different: it is estimated from data and depends on which rows you look at. Those fills are deliberately left out of the cleaning step and done later inside each training fold (Chapter 3).

A technical detail called rule **P01** makes several rules possible. pandas treats the text "NA" and "None" as missing when it reads a CSV. But in this dataset "None" is a valid veneer category and "NA" is a marker meaning "feature absent". The cleaning code therefore reads each file twice, once normally and once with `keep_default_na=False`, and uses the raw copy to tell the cases apart.

### 2.2 The rules by family

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

### 2.3 The C01 story: a reconstruction, not a recovery

In the original notebook the cell that defined the list of columns used by C01 (`c01_cols`) was lost. Cell 65 still referenced it, the execution counters jumped from 59 to 63, and a clean run raised a `NameError`. The code is missing in every available copy, including an older backup. What survives is (a) the written description in the cleaning log and (b) the saved verification output, which says C01 touched 1 entry in 1 house in train and 12 entries in 9 houses in test.

We therefore wrote the rule again from that description and **labelled it as a reconstruction** everywhere (docstring, tests, notebook). The test suite checks that the reconstruction reproduces exactly the documented counts. A reconstruction that matches the saved evidence is good enough to continue, but you must never say "we recovered the code". In an interview this is a good example of honest provenance: say what you know, what you rebuilt and how you verified it.

### 2.4 B04: a condition instead of an Id

The original notebook hard-coded the basement-unknown case as `Id == 2121`. That works for this one file but encodes a fact about a row, not a rule. We replaced it with a condition: all basement fields are missing. The condition selects exactly Id 2121 in test and nothing in train, and a test asserts both. The benefit is that the same rule will behave correctly on any new data, and it is honest about why the house is special.

A related side-effect: the flags `BasementStatusUnknown`, `GarageYearUnknown` and `MiscFeatureConflict` are constant (all False) in training. They are removed from the feature set (`config.CONSTANT_IN_TRAIN_FLAGS`), because a column that never varies cannot teach a model anything. They still document the test rows that need care.

---

## Chapter 3. What a "fitted statistic" is and why it lives inside the fold

### 3.1 The definition

A **fitted statistic** is any number a preprocessing step computes from data and then reuses: a median, a mean, a category list, a scaling range, "which columns are skewed". The test is simple: if the number would change when you show the step a different set of rows, it is a fitted statistic.

Compare two operations.

- "If `PoolArea` is 0 and `PoolQC` is missing, write NoPool." Whatever other rows you show, the answer for this row is the same. Not fitted. May run once, anywhere.
- "Replace a missing `LotFrontage` with the median frontage of this house's neighbourhood." The median depends on which houses you include. Fitted.

### 3.2 Why fitted statistics must live inside the fold

In cross-validation the validation fold plays the role of future, unseen houses. If a median or a skewness is computed using the validation rows, the training procedure has indirectly seen them. The score becomes optimistic: that is **leakage**. The effect for imputation medians is small, but the habit matters, and for target-dependent steps it can be enormous. The discipline is: **every fitted step is refit on each training fold and only applied to the validation fold.** In scikit-learn this is achieved by putting the preprocessing and the model in one `Pipeline`; cross-validation then clones and refits the whole pipeline per fold.

### 3.3 The three fitted components in this project

| Component | What it learns from the training fold | What is applied to unseen rows |
|---|---|---|
| `GroupMedianImputer` | the median `LotFrontage` of each neighbourhood, plus a global median as fallback | fills a missing frontage with its neighbourhood median, or the global median if the neighbourhood is new or all-missing |
| `SkewLog1pTransformer` | which numeric columns have skewness above 0.75 and minimum at least 0 | applies `log1p` to exactly those columns, whatever the unseen rows look like |
| `KNNLotFrontageImputer` | a standardiser plus the training houses (their lot area, floor area, quality, year built, frontage) | fills a missing frontage from the five nearest training houses |

The skew transformer is the subtle one. The choice of columns is a statistic, so it is learned from the training fold. Two folds can therefore pick different columns. The unit tests check exactly that: the selected columns differ between two different samples, and `transform` on unseen rows uses the fitted list and does not re-measure skewness.

The scaler and the one-hot encoder inside `make_preprocessor` are fitted statistics too (medians, interquartile ranges, category lists). Unseen categories at prediction time are ignored by the encoder rather than raising an error.

### 3.4 Why the cleaning rules may run once

The seventeen rules are not fitted. Each decision for a house uses only that house's own columns and the fixed data dictionary. We run them on the full training and test tables once, identically for both. The tests also confirm the rules are idempotent (running them twice changes nothing) and that they never touch `SalePrice`. This is why `clean_train_test` is called once in `data.load_development_data`, while the preprocessor is rebuilt inside every fold.

One more rule of the same family: the **feature groups** of Chapter 4 are pure row-wise formulas, so they may also be built once. Their missing results (for example `GarageAge` for houses with no garage) are left as missing and filled later, inside the fold.

---

## Chapter 4. The engineered feature groups

### 4.1 How the ablation was run

We tested each group by adding it alone to `base` ("base+group"), by adding all five ("all"), and by removing each group in turn from "all" (leave-one-out). Every configuration was scored with Ridge (alpha 10), ElasticNet (alpha 0.001, l1 ratio 0.5) and XGBoost with default settings, using the same five folds. We compare configurations with a **paired** difference: for each fold we subtract the two scores and look at the mean and the standard error (SE) of those five differences. Pairing matters because the folds differ a lot in difficulty (Ridge fold scores range from 0.112 to 0.185 on base). A difference smaller than about one paired SE, and certainly below 0.001, is noise.

Reference scores on base: Ridge 0.14088 (E003), ElasticNet 0.13786 (E005), XGBoost 0.14868 (E004).

### 4.2 Group by group

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

### 4.3 The lesson: quality_interactions carried the signal, ordinal_quality did not

**quality_interactions** is the only group whose removal clearly hurts every model, by 0.0025 to 0.0086, and it also improves every model when added alone. The reason is mechanical. A linear model adds effects: it can say "each quality point is worth x" and "each square foot is worth y" but not "a quality point is worth more in a bigger house". The product column supplies exactly that. XGBoost can in principle build such interactions from splits, but it needs many splits to approximate a smooth product, so a ready-made column helps it most (0.0099).

**ordinal_quality** looked like it should help, because quality labels are ordered, and in the one-hot form the model treats Ex and Gd as unrelated. But those same ten columns are already represented, and the one-hot columns of the same labels stay in the data (the group keeps the text columns). The numeric twin therefore adds a second, partly redundant encoding. Alone it moves Ridge and ElasticNet by +0.0003 and -0.0001, which is noise. Removing it from "all" slightly improves the two linear models (0.0004 and 0.0006) but slightly hurts XGBoost (0.0012). Neither direction exceeds one paired SE for Ridge and XGBoost. The lesson is general: **a feature that re-encodes information the model already has adds nothing, while a feature that gives the model a relationship it cannot build (a product) adds a lot.**

**The others.** areas, baths and ages are individually small for linear models (0.0003 or less), but give XGBoost something: baths helps it by 0.0055. Totals save the trees several splits. At default hyper-parameters this evidence is weak; tuning may change it, which is why the registry decisions are marked provisional (D14).

### 4.4 The ruling D11 and why we did not prune

The best linear results at default alpha came from removing ordinal_quality, and the best XGBoost result from removing ages. We did not prune per model. Every leave-one-out variant lies within about 0.001 of "all" except the removal of quality_interactions. Choosing the winner per model from differences that size, with untuned hyper-parameters and five folds, would be fitting to fold noise. The working set is the full "all" (D11), and the ordinal_quality question is re-examined on the finalists with RepeatedKFold 5x3.

### 4.5 The skew transform: the biggest single gain

The in-fold `log1p` of skewed non-negative numeric columns improved the linear models by about 0.009 and was consistent across feature sets (D12).

| Configuration | Ridge | ElasticNet | Ids |
|---|---|---|---|
| base | 0.14088 | 0.13786 | E003, E005 |
| base with skew | 0.13192 | 0.12845 | E039, E040 |
| all | 0.13871 | 0.13503 | E021, E022 |
| all with skew | 0.13042 | 0.12562 | E041, E042 |

The gain is larger than any feature group (0.0083 to 0.0094 versus at most 0.0029 for the best group). The mechanism: a column like `LotArea` has skewness 13 because a few giant lots exist; after `log1p` it is about -0.3. A linear fit is then no longer controlled by those few lots. Trees only use the order of values, and `log1p` does not change order, so XGBoost does not receive the transform.

### 4.6 LotFrontage: a non-issue

`LotFrontage` is missing for about 17% of houses. Neighbourhood median (E043, E046), global median (E044, E047) and KNN (E045, E048) all land within noise.

| Method | Ridge | XGBoost |
|---|---|---|
| neighbourhood median | 0.14088 | 0.14868 |
| global median | 0.14099 | 0.14701 |
| KNN | 0.14048 | 0.14900 |

Ridge differences are at most 0.0005. XGBoost's global median is 0.0017 better than the neighbourhood median, within one paired SE. We keep the neighbourhood median (D13), the documented choice of the original cleaning log, because nothing more elaborate beats it. The general lesson: when a column carries little signal after other columns are present (here lot area already proxies frontage), a cleverer fill rarely moves the score.

---

## Chapter 5. The outlier-policy lesson (D15)

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

## Chapter 6. The anomaly flags, and test-only anomalies at prediction time

### 6.1 The finding

Three audit flags mark impossible years: `GarageYrBltImplausible` (garage built after the sale), `SoldBeforeBuilt` and `RemodBeforeBuilt`. They are produced by `cleaning.add_anomaly_flags`, which only adds columns.

| Flag | Train | Test | Test Id |
|---|---|---|---|
| GarageYrBltImplausible | 0 | 2 | 2593 (garage year 2207), 2550 |
| SoldBeforeBuilt | 0 | 1 | 2550 (sold 2007, built 2008) |
| RemodBeforeBuilt | 0 | 1 | 1877 (remodelled 2001, built 2002) |

All three flags are zero for every training house. They are non-zero only for three test houses. A column that is constant in the training data cannot be learned from, so `base+anomaly_flags` can never differ from base. We kept the group in `FEATURE_SETS` for documentation but excluded it from the default ablation to save budget. This is a real data finding: **the training data contain no examples of this kind of anomaly, so no model trained on it can know what to do with them.**

### 6.2 What to do at prediction time

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

## Self-check questions

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

## Answers

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
