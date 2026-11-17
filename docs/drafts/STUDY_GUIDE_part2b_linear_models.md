DRAFT — status: complete (sections done: 1 OLS and collinearity, 2 Ridge, 3 Lasso, 4 ElasticNet, 5 skew correction, 6 scalers, 7 convergence and cost, 8 grid results E055-E057, 9 ensemble partners, 10 self-check questions with answers; remaining: none, pending Agent B review)

# Study guide, part 2b: the linear models

Draft by Agent A (feature-researcher) for the director to merge into `docs/STUDY_GUIDE.md`. Agent B will review the modelling content. Same audience as parts 1 and 2a. Numbers come from the registry, `docs/EXPERIMENT_HISTORY.md`, the grid file `experiments/models/linear_grid_base+areas+baths+ages+quality_interactions+ordinal_quality_keep_all.csv`, the diversity file `experiments/ensemble/diversity_20261113_121446.csv`, and two documentation measurements made for this chapter (marked as such). All errors are RMSE on log1p(SalePrice), five shuffled folds, seed 42, lower is better.

Notation. X is the design matrix with n rows and p columns, y the log target, beta the coefficient vector. After preprocessing, p = 352 columns for the "all" feature set (53 numeric, about 299 one-hot levels), and each training fold has about 993 rows.

---

## 1. Ordinary least squares and why collinearity breaks it

### 1.1 The estimator

OLS chooses beta to minimise the sum of squared residuals, ||y - X beta||^2. Setting the gradient to zero gives the normal equations and the closed form

beta_hat = (X'X)^(-1) X'y.

Under the standard model y = X beta + e with noise variance sigma^2, the estimator has covariance

Var(beta_hat) = sigma^2 (X'X)^(-1).

### 1.2 What collinearity does

Write the eigendecomposition of X'X with eigenvalues lambda_1 >= ... >= lambda_p. Then (X'X)^(-1) has eigenvalues 1/lambda_j. If two columns are nearly proportional (or one is nearly a combination of others), the smallest eigenvalue is close to zero, and the variance of the coefficient along that direction is about sigma^2 / lambda_min, which explodes. The fitted values can stay fine, because the data pin down the *sum* of the collinear effects. But the individual coefficients become huge with opposite signs, and a small change in the training rows swings them wildly. When a validation house has a slightly different mix of the collinear columns, the prediction swings too.

### 1.3 The Ames data are exactly collinear

This is not a hypothetical. In the base data two exact identities hold on every house (a documentation measurement made for this chapter):

- `GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF`
- `TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF`

With 35 numeric columns plus an intercept, the design matrix has rank 34, not 36, and its condition number is about 7e15, which is numerically singular. The engineered `TotalSF` (basement plus two floors) adds yet another exact dependency. One-hot encoding adds the "dummy variable trap": the dummies of one categorical column sum to one, which equals the intercept column.

Plain OLS (E002, numeric columns only) scores 0.1599 with fold scores 0.134, 0.217, 0.168, 0.136, 0.144 and a fold standard deviation of 0.0348. The 0.217 in fold 2 is what an unstable fit looks like: one fold with a different mix of collinear columns and the hard houses of that fold gives a far larger error than the others. We did not diagnose that fold further, so treat collinearity as the likely, not proven, cause. Compare Ridge on the same folds with far more columns (E003): 0.1409 with fold std 0.0316, and the tuned penalised models reach a fold std of about 0.023. Regularisation does not only improve the mean, it removes instability.

---

## 2. Ridge regression (L2)

### 2.1 Objective and closed form

Ridge adds a squared-length penalty. In scikit-learn the objective is

minimise ||y - X beta||^2 + alpha ||beta||_2^2

(note: no 1/n in front of the data term). Setting the gradient to zero gives

beta_ridge = (X'X + alpha I)^(-1) X'y.

Adding alpha to the diagonal lifts every eigenvalue of X'X to lambda_j + alpha, so the matrix is always invertible and the variance along the flat direction is bounded by sigma^2 / alpha. Using the singular value decomposition X = U D V', the fitted coefficient along direction j is shrunk by the factor d_j^2 / (d_j^2 + alpha): well-determined directions (large d_j) are barely touched, poorly determined ones (small d_j) are shrunk strongly toward zero. This is exactly the right behaviour for collinear columns: the unstable direction is suppressed and the stable one kept.

Ridge has a bias-variance reading. alpha = 0 is OLS (no bias, high variance). As alpha grows, variance falls and bias rises; the CV error is U-shaped in alpha.

### 2.2 Shrinkage of one-hot levels

For a one-hot column of a rare category (say 10 houses), the data say little, and the least-squares coefficient is noisy. Ridge shrinks it toward zero, which means "this rare level behaves like the baseline category". Ridge never sets a coefficient exactly to zero: in the fitted E055 pipeline 351 of 352 coefficients are non-zero (a documentation measurement). It can only make small ones smaller.

### 2.3 The flat alpha curve around 10

From the grid file (feature set "all", skew correction on, RobustScaler):

| alpha | 1 | 3.2 | 5.6 | 10 | 17.8 | 31.6 | 100 |
|---|---|---|---|---|---|---|---|
| cv_mean | 0.13473 | 0.13137 | 0.13064 | 0.13042 | 0.13067 | 0.13134 | 0.13395 |

The minimum is at alpha = 10, and moving alpha by a factor of about 2 in either direction (5.6 or 17.8) costs only 0.0002 to 0.0003. That is well inside the 0.003 noise band (part 2a). The consequence, recorded as D16, is that further tuning of Ridge cannot close its gap: the curve is flat, and the remaining 0.005 difference to the L1 family is structural, not a tuning problem. The alphas of Ridge and Lasso must not be compared directly: Ridge's objective has no 1/n factor, so Ridge's alpha = 10 with about 993 rows corresponds to roughly 10/993 = 0.01 on the scale where Lasso's alpha lives.

---

## 3. Lasso (L1)

### 3.1 Objective

Scikit-learn's Lasso minimises

(1 / (2n)) ||y - X beta||^2 + alpha ||beta||_1, where ||beta||_1 = sum_j |beta_j|.

There is no closed form, because |beta_j| is not differentiable at zero, but the problem is convex and has a unique solution on well-behaved data.

### 3.2 Soft-thresholding and sparsity

The simplest case shows the mechanism. If the columns are orthonormal in the sense X'X / n = I, each coefficient decouples and the Lasso solution is the **soft-threshold** of the OLS coefficient:

beta_lasso_j = sign(beta_ols_j) * max(|beta_ols_j| - alpha, 0).

Coefficients smaller than alpha in absolute value are set **exactly to zero**; larger ones are shrunk by alpha. (Ridge, in the same setting and with a penalty alpha ||beta||^2 next to the same 1/(2n) data term, multiplies every coefficient by 1 / (1 + 2 alpha) and zeroes none.) Geometrically, the L1 ball has corners on the axes, so the constrained optimum tends to land on a corner, where some coefficients are zero. This is why Lasso performs variable selection.

For correlated columns the effect is the same in spirit: Lasso keeps one column of a correlated group and zeroes the others, but which one it keeps is somewhat arbitrary and can change between folds.

### 3.3 How many coefficients survive at the optimum (documentation measurement)

We fitted the E056 pipeline (Lasso, alpha = 5.62e-4, RobustScaler, skew correction, feature set "all") on the 1,241 development rows:

| Quantity | Value |
|---|---|
| Columns after preprocessing | 352 |
| Non-zero coefficients | **115** |
| Zero coefficients | 237 (67%) |
| One-hot columns | 299, of which 82 are non-zero |

So Lasso discards two thirds of the columns, mostly rare category levels. The largest coefficients show a caution you should know. The biggest is `RoofMatl_ClyTile` at -0.60, and in the development rows that level occurs in exactly **one house**; `MSZoning_C (all)` (10 houses) has -0.30 and `Condition2_PosN` (2 houses) -0.17. A coefficient for a single house is simply that house's residual absorbed by a dummy column: it makes the training fit better and is a pure memorisation, which the penalty has not removed because the gain in fit exceeds the penalty at this small alpha. It does not generalise, and it is harmless only because that level is almost never seen again. The meaningful coefficients are those of `Neighborhood` levels (Crawfor, StoneBr, NoRidge, NridgHt, Edwards), `GrLivArea`, `OverallQual`, `TotalSF` and `HouseAge`.

### 3.4 Why the optimum near 5.6e-4 is sharp

From the grid file (RobustScaler):

| alpha | 1.0e-4 | 3.2e-4 | 5.6e-4 | 1.0e-3 | 1.8e-3 | 3.2e-3 |
|---|---|---|---|---|---|---|
| cv_mean | 0.13397 | 0.12780 | **0.12545** | 0.12650 | 0.13150 | 0.13673 |

Compared with Ridge's flat bowl, Lasso loses 0.0024 by halving alpha and 0.0010 by doubling it (E056 sits at 5.62e-4). The reason is that alpha controls two things at once. It shrinks every surviving coefficient, and it decides *which* coefficients are zero. When alpha is a bit too small, noisy rare levels and redundant columns enter and variance grows. When alpha is too large, genuine signal columns drop below the threshold and are removed, so bias grows quickly. With about 352 mostly weak columns the model sits on a ridge between these two failure modes. Because the optimum is sharp and was found on the same folds that selected it, it carries a selection optimism of about +0.0027 (Agent D's measurement: 0.1255 on the seed-42 folds, 0.1282 on seed-7 folds).

---

## 4. ElasticNet

### 4.1 Objective

ElasticNet mixes the two penalties with a ratio rho (`l1_ratio`):

(1 / (2n)) ||y - X beta||^2 + alpha * rho * ||beta||_1 + (alpha * (1 - rho) / 2) * ||beta||_2^2.

rho = 1 is the Lasso; rho near 0 is almost Ridge. The L2 part makes the problem strictly convex, so correlated columns tend to share weight (the "grouping effect") instead of one being picked arbitrarily.

### 4.2 Why l1_ratio between 0.5 and 0.9 ties

The best cv_mean per l1_ratio in the grid is: 0.1 gives 0.12687, 0.3 gives 0.12622, 0.5 gives 0.12555, 0.7 gives 0.12594, 0.9 gives 0.12581, and the pure Lasso gives 0.12545. Between 0.5 and 0.9 the differences are below 0.0004, far inside the noise band. There is a simple reason in the numbers: the registered ElasticNet (E057) has alpha = 1.18e-3 and rho = 0.5, so its effective L1 weight is alpha times rho, about 5.9e-4, which is practically the Lasso optimum of 5.6e-4. The two models are asked to zero out the same columns by the same amount; the extra L2 term, which has a small effect when most columns are one-hot dummies that are not strongly correlated with each other, only smooths the solution. Hence the Lasso and ElasticNet residuals correlate at 0.997 (section 8).

---

## 5. Why skew correction helps linear models by about 0.009 but not trees

The in-fold `SkewLog1pTransformer` (part 1, section 4.5) improved Ridge and ElasticNet by 0.0083 to 0.0094 in the ablation (E039 to E042) and is switched on for all linear models (D12). The reasons are specific to what a linear model assumes.

1. **A linear model assumes a straight line between each input and the (log) price.** For right-skewed inputs the relationship with log-price is usually closer to a straight line in the *logarithm* of the input than in the raw input. (Price per square foot is roughly constant, so log-price is linear in log-area.) After `log1p`, the straight-line assumption is a better fit.
2. **Leverage.** In least squares, observations with extreme input values have large leverage and pull the line toward themselves. `LotArea` has a skewness of 13 in development because of a few giant lots; after `log1p` it is about -0.3. Without the transform a handful of houses control the `LotArea` coefficient.
3. **Penalties depend on scale.** A penalised model charges for coefficient size, and a column dominated by a few huge values has most of its variation in the tails, so its coefficient is determined by those few rows.

**Trees do not benefit** because they split on thresholds: a tree only uses the *order* of values in a column, and `log1p` is strictly increasing, so every split is the same before and after. A transform that does not change order cannot change a tree, so applying it would only add cost (D12).

---

## 6. RobustScaler versus StandardScaler

StandardScaler subtracts the mean and divides by the standard deviation. RobustScaler subtracts the median and divides by the interquartile range (the 25th to 75th percentile), so extreme values do not distort it. Scaling matters for penalised models, since the penalty treats all coefficients equally and the units of a column change its coefficient: a column in square feet needs a tiny coefficient and a column in a 0/1 flag a large one.

The grid shows it **does not matter here**: for every family the best robust and standard scores differ by at most 0.0001 (Ridge 0.13042 vs 0.13028; Lasso 0.12545 vs 0.12556; ElasticNet 0.12555 vs 0.12556 at l1_ratio 0.5). The reasons: after the skew transform most numeric columns are close to symmetric, so mean and median, and standard deviation and IQR, are similar. For the 0/1 dummies the two scalers differ in principle (RobustScaler leaves a dummy whose IQR is 0 unscaled, StandardScaler divides by sqrt(p(1-p)), making rare dummies larger and therefore cheaper to penalise), but those columns are a minority of the signal. We register Ridge with the standard scaler and the L1 models with the robust one only because those were the grid winners; the choice is not a finding.

---

## 7. Convergence, `max_iter` and cost

### 7.1 Coordinate descent

Lasso and ElasticNet are solved by **coordinate descent**: cycle through the columns, and for each j update beta_j in closed form while holding the others fixed. For the Lasso the update is a soft-threshold:

beta_j <- S( rho_j, alpha ) / z_j, with rho_j = (1/n) x_j' r_j, z_j = (1/n) ||x_j||^2,

where r_j is the residual with the contribution of column j added back. One sweep over all columns costs O(n * p), here about 993 * 352, roughly 350,000 multiply-adds, a fraction of a millisecond. The algorithm stops when the *duality gap* (a certified bound on the distance from the optimum) falls below `tol` times the scale of the target, or when `max_iter` sweeps are reached.

### 7.2 max_iter and the one warning

The model builders in `houseprices.models` set `max_iter = 20000` (scikit-learn's default is 1000). Convergence is slowest when alpha is small and columns are strongly correlated, because the penalty then barely constrains the problem and coordinate descent zig-zags along flat valleys; each sweep makes only a small step. Of the 136 grid fits, exactly **one** raised a `ConvergenceWarning`: ElasticNet with alpha 1e-4 and l1_ratio 0.1 under the standard scaler, that is, the weakest regularisation with the least L1 in the grid. It is far from the optimum (E057 uses an alpha about 12 times larger) and therefore has no effect on any registered result. If you see a ConvergenceWarning near the optimum, the first fixes are a larger `max_iter` or a larger alpha. Ridge has no such issue, since it is solved in closed form (or by a direct linear-algebra solver).

### 7.3 Cost in practice

Each five-fold CV of a linear model in this project takes about one second, which is why 136 grid points ran in a few minutes. It is the reason the linear family can be searched exhaustively while the boosters needed Optuna.

---

## 8. What the grid results say (E055 to E057)

| Id | Model | Hyper-parameters | Scaler | cv_mean | Fold std |
|---|---|---|---|---|---|
| E055 | Ridge | alpha = 10 | standard | 0.13028 | 0.0232 |
| E056 | Lasso | alpha = 5.62e-4 | robust | 0.12545 | 0.0235 |
| E057 | ElasticNet | alpha = 1.18e-3, l1_ratio = 0.5 | robust | 0.12555 | 0.0230 |

Reading, with the necessary caution:

- The L1 family beats Ridge in **every** fold, by 0.005 on average (per-fold gaps 0.003 to 0.007), which is the strongest kind of evidence available with five folds (a consistent paired difference). The explanation offered in the history file is that with about 350 mostly one-hot columns and 993 training rows, zeroing rare levels works better than shrinking them. Our fitted Lasso keeps 115 of 352 columns.
- Lasso and ElasticNet tie (0.0001 apart). Pick either; D16 keeps both as finalists but they are nearly the same model.
- Fold 2, which holds Id 524, remains the hardest fold (0.155 to 0.160) for every linear model.
- The registered bests are the minima of 136 comparisons on the same folds, so they are optimistic by about 0.002 to 0.003 (part 2a). The finalist check on fresh seed-7 folds will state the honest number.

---

## 9. Linear models as ensemble partners

A linear model and a boosted tree make different mistakes: the linear model is smooth and additive, the tree is piecewise constant and finds interactions. Part 3 covers stacking and blending; the evidence for diversity is in the diversity file (`experiments/ensemble/diversity_20261113_121446.csv`, single-pass out-of-fold predictions, used only for this diagnostic).

| Pair type | Residual correlation | Prediction correlation |
|---|---|---|
| linear with linear (Ridge, Lasso, ElasticNet) | 0.970 to 0.997 | 0.996 to 1.000 |
| tree with tree (XGBoost, LightGBM, CatBoost) | 0.946 to 0.962 | 0.994 to 0.996 |
| linear with tree | **0.823 to 0.862** | 0.980 to 0.985 |

The maths of why this matters: take two predictors whose errors have equal variance s^2 and correlation rho. The error of their equal average has variance s^2 (1 + rho) / 2, so its RMSE is s * sqrt((1 + rho) / 2). For rho = 0.85 (linear with tree) that is 0.961 s, about 4% better than either model; for rho = 0.97 (two linear models) it is 0.992 s, under 1%. A third near-duplicate adds almost nothing. This is why the observed gain comes from one linear plus one tree, not from many linear models: the first registered ensemble, an equal average of Lasso and CatBoost (E061, Stage 1 bases), scored 0.11807 against 0.11946 for CatBoost alone (-0.0014, inside the 0.003 noise band; with the final Stage 2 CatBoost the gain shrank to -0.0007, and on the fresh finalist folds the average was 0.0010 better than CatBoost alone, a third of a standard error, see parts 3 and 4), while a blend of the three linear models (E064) was no better than Lasso alone.

Note that the prediction correlations are all 0.98 or higher because the models predict the same target; it is the correlation of the *errors* that decides how much averaging helps.

---

## Self-check questions

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

## Answers

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
