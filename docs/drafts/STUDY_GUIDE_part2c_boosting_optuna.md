DRAFT - status: complete except section 8 (Stage 2 results, placeholders marked `<<STAGE2: ...>>` for the director). Sections: 1 decision trees, 2 gradient boosting, 3 XGBoost / LightGBM / CatBoost, 4 Optuna and TPE, 5 the staged budget and the Stage 1 table, 6 the real code, 7 verification notes, 8 Stage 2 results, 9 self-check questions with answers.

# Study guide, part 2c: boosted trees and Optuna tuning

Draft by Agent B (model-researcher) for the director to merge into `docs/STUDY_GUIDE.md`. Same audience as parts 1 to 2b. All errors are RMSE on log1p(SalePrice), five shuffled folds (seed 42), 1,241 development rows, about 993 training rows per fold, lower is better. Numbers come from `docs/EXPERIMENT_HISTORY.md`, `experiments/tuning/stage1_summary.json`, the per-trial CSVs `experiments/tuning/stage1_*_trials.csv` (only trials 0 to 24 are Stage 1; the files are rewritten when Stage 2 ends), the source files `src/houseprices/models.py` and `experiments/tuning/run_optuna_stage1.py`, and three small checks run for this chapter (marked "documentation measurement"). The toy examples with five houses are invented for illustration and are labelled as such.

---

## 1. Decision trees for regression

### 1.1 What a tree is and how it splits

A regression tree partitions the rows into boxes by asking threshold questions ("OverallQual <= 5?") and predicts the **mean of y** inside each box (each leaf). To choose a split the tree tries every feature j and every threshold t between consecutive sorted values, and keeps the pair that minimises the total sum of squared errors of the two children:

SSE(j, t) = sum over i in left of (y_i - mean_L)^2 + sum over i in right of (y_i - mean_R)^2.

Equivalently it maximises the reduction in variance. The same rule is then applied recursively inside each child until a stopping rule (maximum depth, minimum rows per leaf, minimum gain) is hit.

Toy example (invented numbers). Five houses with log prices y = 11.8, 11.9, 12.0, 12.6, 12.7 and OverallQual x = 4, 5, 5, 8, 9. The mean is 12.2 and the parent SSE is 0.16 + 0.09 + 0.04 + 0.16 + 0.25 = 0.70. The split "x <= 5" gives left {11.8, 11.9, 12.0} with mean 11.9 and SSE 0.02, and right {12.6, 12.7} with mean 12.65 and SSE 0.005. Total 0.025, a reduction of 0.675. The split "x <= 4" gives left {11.8} (SSE 0) and right {11.9, 12.0, 12.6, 12.7} with mean 12.3 and SSE 0.50, so it is far worse. The algorithm is greedy: it picks the best split now and never looks ahead.

A single deep tree has low bias and very high variance: it can memorise the 993 training rows and a small change in the data changes the tree. Boosting (section 2) fixes this by combining many **shallow** trees.

### 1.2 Invariance to monotone feature transforms, and ruling D12

Look at what the split search uses: only the **order** of the values in a column. If g is any strictly increasing function (log1p, square root, a rank transform) the sorted order of g(x) equals that of x, so the best partition of the training rows is the same; only the numeric value of the threshold changes (it is placed between the same two neighbours). Hence a tree fitted on log1p(LotArea) predicts exactly like one fitted on LotArea on the training rows. For unseen rows the two can differ only if a value falls between two neighbouring training values on the wrong side of a re-placed threshold, which is negligible. XGBoost and LightGBM first bin each feature into quantile-based buckets, and quantile bins also depend only on the order, so the invariance holds for them too.

This is the basis of ruling **D12**: the in-fold `SkewLog1pTransformer` is applied to Ridge, Lasso and ElasticNet (it gained about 0.009 for them, part 2b section 5) and never to the trees, because for a tree it cannot change the model and would only add computation. The boosters therefore use `TREE_PREPROCESSOR_DEFAULTS = {"scale": False}` in `models.py`: no scaling and no skew transform.

Two cautions so you do not over-generalise. First, invariance applies to **features**, not to the **target**: the target is log1p(SalePrice), and a tree does care about it because the SSE is computed on y itself. Training on the log target is what makes the boosters optimise the same quantity the competition scores. Second, one-hot columns are still needed here: the preprocessor gives the trees 0/1 columns (299 of 352 columns) for categories; only CatBoost with native categoricals (section 3.3) consumes the raw text.

---

## 2. Gradient boosting

### 2.1 The additive model

Boosting builds the predictor as a sum of small trees:

F_M(x) = F_0 + eta * h_1(x) + eta * h_2(x) + ... + eta * h_M(x),

where F_0 is a constant (the mean of y), each h_m is a shallow tree, and eta is the learning rate (shrinkage). Trees are added one at a time and earlier trees are never changed.

### 2.2 Functional gradient descent

Choose a differentiable loss L(y, F). Ordinary gradient descent updates parameters in the direction of the negative gradient. Boosting does the same in **function space**: at round m, compute for every training row the negative gradient of the loss with respect to the current prediction,

r_i = - dL(y_i, F)/dF evaluated at F = F_{m-1}(x_i),

then fit a tree h_m to these pseudo-residuals r_i and add eta * h_m. The tree is a cheap, smooth-enough approximation of "move every prediction downhill".

For squared error, L = (1/2)(y - F)^2, the negative gradient is exactly the ordinary residual y_i - F_{m-1}(x_i). So squared-error boosting is literally "fit a tree to the residuals, add a fraction of it, repeat". Our target is already log1p(SalePrice), so squared error on it is the RMSE the project reports.

Continue the toy example. F_0 = 12.2, residuals -0.4, -0.3, -0.2, +0.4, +0.5. A tree with the split "x <= 5" predicts the leaf mean of the residuals: -0.3 on the left and +0.45 on the right. With eta = 0.1 the new predictions are 12.2 + 0.1 * (-0.3) = 12.17 for the first three houses and 12.2 + 0.1 * 0.45 = 12.245 for the other two, so the new residuals are -0.37, -0.27, -0.17, +0.355, +0.455. Each round removes only 10 percent of what the leaf can explain, and the next tree is free to choose a different split on the new residuals.

### 2.3 Learning rate versus number of trees

For a fixed leaf, the remaining error in that leaf shrinks like (1 - eta)^m. With the tuned CatBoost rate eta = 0.0347 and 1,700 rounds, (1 - 0.0347)^1700 is about e^(-60), so the part of the training error the trees can express is essentially gone. Two consequences. First, a small eta needs a large M to reach the same fit: the product eta * M is roughly the total distance travelled (about 60 for the tuned XGBoost 0.0392 x 1600 and CatBoost 0.0347 x 1700; 16.5 for LightGBM 0.0103 x 1600, a different tree shape). Second, small steps act as regularisation: many tiny corrections average out the noise that one big step would copy. That is why the tuned boosters are "slow" (section 5) and why the number of trees is a tuned hyper-parameter here rather than chosen by early stopping. This project does not use early stopping because it would need a validation set inside each training fold; the whole rounds count is tuned in the outer cross-validation instead.

### 2.4 Subsampling and column sampling

Stochastic gradient boosting (Friedman) fits each tree on a random fraction `subsample` of the rows, drawn without replacement, and many libraries also sample columns. Both inject randomness so that successive trees are less correlated and the sum has lower variance; they also make each round cheaper. The tuned values are subsample 0.76 (XGBoost), 0.76 (LightGBM) and 0.62 (CatBoost), with column fractions 0.80, 0.89 and `rsm` 0.79. On 993 rows a tree sees about 600 to 750 of them per round, a visible dose of noise that the cross-validation judged helpful.

---

## 3. The three libraries

### 3.1 XGBoost: second-order objective, gain, regularisation

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

### 3.2 LightGBM: leaf-wise growth, histograms

LightGBM grows trees **leaf-wise** (best-first): at every step it splits whichever current leaf, anywhere in the tree, has the largest gain, until `num_leaves` leaves exist. A level-wise tree wastes splits on leaves that gain little; leaf-wise reaches lower loss for the same number of leaves, but can grow lopsided, deep branches that fit noise. That is why `num_leaves`, not depth, is the main complexity knob (default `max_depth = -1`, unlimited, checked in the installed library), and why `min_child_samples` (rows per leaf, default 20) matters: it stops a lopsided branch from isolating two or three houses. The space allows 5 to 40; the tuned value is 7, below the default of 20.

Training uses **histograms**: each feature is bucketed once into a limited number of bins (255 by default; library default recalled, not checked here), and a split search scans the bins' gradient sums instead of the sorted rows, with the trick that a child's histogram equals its parent's minus its sibling's. On 993 rows with mostly 0/1 columns the speed gain is small and the binning changes nothing material, but it is the reason LightGBM is the fastest of the three (1.7 min for 25 trials).

**`subsample_freq`.** In LightGBM, row subsampling ("bagging") happens only every `subsample_freq` iterations, and the default is 0, which disables it: `subsample = 0.76` alone would be silently ignored. We confirmed the library defaults (`subsample_freq` 0, `subsample` 1.0). The search space therefore fixes `subsample_freq=1` (bag every iteration), with the comment in `models.py` and an `assert kwargs["subsample_freq"] == 1` when registering. XGBoost and CatBoost do not have this trap.

### 3.3 CatBoost: symmetric trees, ordered statistics

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

## 4. Hyper-parameter tuning with Optuna

### 4.1 Why not a grid

The XGBoost space has 8 parameters and the full grid would be astronomically large, and most of its volume is useless (for example depth 8 with 500 rounds). Random search covers important dimensions better than a grid but ignores what it has learned. TPE uses the history of trials to decide where to try next.

### 4.2 What TPE does

TPE (Tree-structured Parzen Estimator) models p(x | y) instead of p(y | x). After n completed trials it sorts them by objective value and splits them at a quantile gamma (not XGBoost's gamma): the best gamma-fraction is the "good" set and the rest the "bad" set. It fits a density l(x) to the hyper-parameters of the good trials and g(x) to the bad ones (Parzen estimators, that is kernel density mixtures). Then p(x | y) is l(x) if y is below the quantile threshold y* and g(x) otherwise, and the expected improvement over y* turns out to be proportional to

(gamma + (1 - gamma) * g(x) / l(x))^(-1),

so **maximising l(x) / g(x)** is the same as maximising expected improvement: choose settings that are common among good trials and rare among bad ones. In practice Optuna draws 24 candidates from l(x) (`n_ei_candidates = 24`) and keeps the one with the highest ratio.

Defaults read from the installed Optuna 5.0.0 source: `n_startup_trials = 10` (the first 10 trials are pure random sampling, since there is nothing to model yet), and the good-set size is `min(ceil(0.1 * n), 25)`: 2 good trials when n = 20, 3 at n = 25, 8 at n = 75. With this version the sampler uses the multivariate variant for single-objective studies by default, so l and g are joint densities over all parameters rather than products of independent ones.

The effect is visible in our data (documentation measurement). The mean objective of trials 10 to 24 is clearly below that of the random trials 0 to 9: XGBoost 0.1254 versus 0.1322, LightGBM 0.1290 versus 0.1340, CatBoost 0.1217 versus 0.1240. The search concentrates in good regions. (XGBoost's overall best, 0.12307, was drawn at random in trial 3; TPE's trial 11 reached 0.1231 as well.)

### 4.3 Seeded sampler, SQLite, resumability

`optuna.samplers.TPESampler(seed=42)` makes the proposals reproducible, and the study is stored in `experiments/tuning/optuna_stage1.db` (SQLite, git-ignored) with `load_if_exists=True`. The study name encodes the label, feature set and row filter, so each combination has its own history. On every call the script counts the COMPLETE trials and runs only `n_trials - done` more. This is how Stage 2 works: calling the same command with `--n-trials 75` runs 50 more trials on top of the 25 already stored, and the TPE proposals are conditioned on all of them. A subtlety: the seeded random generator is created afresh at each call and its state is not stored, so a resumed study is a valid TPE run but not bit-for-bit what an uninterrupted 75-trial run would have produced (my reasoning from the code, not tested). A time cap (`--time-cap-minutes`) works through a callback that calls `study.stop()` once the soft deadline has passed; the running trial finishes first, which is why the native study took 27.97 minutes under a 25 minute cap.

### 4.4 Why the objective is the mean of the five selection folds

The objective returned by a trial is `r.mean`: the mean RMSE over the five folds of `KFold(5, shuffle=True, random_state=42)` (`_common.run_cv`, `validation.make_cv`). Averaging over five folds uses every development row for validation exactly once and is much less noisy than one split. Every trial sees the **same** five partitions, so the differences between trials are paired comparisons (the noise of "which houses landed in which fold" cancels), which helps TPE rank configurations. The price is the next subsection.

### 4.5 Selection optimism

If you take the best of N noisy estimates you keep the luckiest one. Suppose N = 25 configurations were equally good and each score had noise sigma, the expected minimum would be about 1.97 sigma below the truth. The seed-to-seed standard deviation of a five-fold CV mean was measured at about 0.0026 (audit), so the worst case is about 0.005. Real configurations differ and share folds, so the measured optimism is smaller: the audit found +0.0020 for a 15-point Ridge grid and +0.0027 for Lasso E056 (0.1255 on the selection folds against 0.1282 on fresh seed-7 folds); the history records roughly 0.002 to 0.005 for best-of-25 boosters. This gives the standing rule: **differences below 0.003 are not wins**, and every score in the registry before the finalist step is preliminary. D17 is the repair: finalists are re-scored with `RepeatedKFold(5 x 3)` on `CV_SEED_FINALISTS = 7`, partitions that share no fold with the seed-42 ones (validation overlap at most 25 percent against 20 percent by chance), so the estimate for the chosen configuration is not selection-biased.

---

## 5. The staged budget and the Stage 1 results

### 5.1 The rule

User correction 7 of the approved plan, ruling D19, and the constants at the top of `run_optuna_stage1.py` (`STAGE2_GAP = 0.003`, `STAGE2_WINDOW = 10`, `IMPROVE_TOL = 1e-4`, `TUNING_CAP_HOURS = 3.0`) define the staged budget:

1. **Stage 1:** 25 trials per booster. Cheap and enough to see the landscape.
2. **Stage 2:** continue the same study to 75 trials only for a booster that is within 0.003 of the best model, **or** whose best-so-far still dropped by more than 1e-4 across the last 10 trials.
3. A total tuning cap of 3 hours (recorded in the summary file; the code does not enforce it, audit F22). Used so far: 10.8 min (Stage 1) plus 28.0 min (E065) = 38.8 min.

The point of staging is to spend compute where the curve is still moving. D19 applied it: XGBoost no (gap 0.0036 and flat since trial 4), LightGBM yes (gap 0.0050 but still improving), CatBoost yes (best and still improving), CatBoost native got its own Stage 1.

### 5.2 The Stage 1 table

| id | model | CV RMSE | fold std | minutes | trial of best | median trial | worst trial | trials within 0.003 of best |
|---|---|---|---|---|---|---|---|---|
| E058 | XGBoost | 0.12307 | 0.0191 | 2.18 | 3 | 0.12641 | 0.15477 | 11 of 25 |
| E059 | LightGBM | 0.12442 | 0.0191 | 1.66 | 24 | 0.12921 | 0.15931 | 5 of 25 |
| E060 | CatBoost | 0.11946 | 0.0170 | 6.93 | 21 | 0.12133 | 0.12975 | 15 of 25 |

(The last four columns are documentation measurements from the trial CSVs, trials 0 to 24.) Best hyper-parameters, from `stage1_summary.json`:

- E058: learning_rate 0.0392, n_estimators 1600, max_depth 3, subsample 0.757, colsample_bytree 0.796, reg_alpha 0.0015, reg_lambda 0.269, min_child_weight 4.
- E059: learning_rate 0.0103, n_estimators 1600, num_leaves 12, subsample 0.759 (subsample_freq 1), colsample_bytree 0.889, reg_alpha 0.143, reg_lambda 0.371, min_child_samples 7.
- E060: learning_rate 0.0347, iterations 1700, depth 5, l2_leaf_reg 2.13, subsample 0.619, rsm 0.788.

### 5.3 What the numbers say

**Shallow, slow, subsampled.** Eight leaves for XGBoost (depth 3), 12 for LightGBM, 32 for CatBoost (depth 5, symmetric); learning rates 0.01 to 0.04 with 1,600 to 1,700 rounds; row subsampling 0.62 to 0.76. This matches the situation: 993 training rows, a noisy target, mostly low-order interactions. Deep trees would memorise; many small, noisy steps average out. The average XGBoost leaf holds about 90 rows (roughly 750 sampled rows spread over 8 leaves).

**CatBoost is more forgiving.** The standard deviation of the 25 trial scores is 0.0028 for CatBoost against 0.0068 (XGBoost) and 0.0070 (LightGBM); 21 of its 25 trials beat 0.125 (9 of 25 for XGBoost, 2 for LightGBM). Even CatBoost's **median** trial (0.12133) beats the best XGBoost trial (0.12307) and the best LightGBM trial. Symmetric trees and its defaults are strong regularisers, so there is less to tune and much less to lose from a bad draw. That is also why E060 stays ahead after the 0.003 optimism correction: the gap to XGBoost is 0.0036.

**What the search found** (Spearman rank correlation of each parameter with the score over 25 trials, positive meaning larger values are worse; with 25 points only |rho| above about 0.4 is worth mentioning, so this is a hint, not a result). LightGBM: `num_leaves` +0.64 (best five trials have a median of 12 leaves, worst five 24) and `min_child_samples` +0.43; large `reg_alpha` hurt (the worst trials had alpha 2.5 to 8.9). XGBoost: `min_child_weight` +0.46, depth effect weak (+0.15) but the best five have a median depth of 3 and the worst five of 6. CatBoost: `l2_leaf_reg` +0.45 (best five median 2.1, worst five 9.5), `iterations` -0.42 (best five 1,700, worst five 600), depth about 0.

**Search-space edges.** The LightGBM best learning rate 0.0103 sits against the lower bound 0.01, and its `min_child_samples` 7 near the bound 5, so the space may be cutting off a better region (more rounds at a still lower rate would need `n_estimators` above 2000). XGBoost's reg_alpha is also near its floor. I did not change anything: it is a point to know when reading Stage 2.

---

## 6. The real code

### 6.1 `src/houseprices/models.py`

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

### 6.2 `experiments/tuning/run_optuna_stage1.py`

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

## 7. Verification notes: things I noticed (for the director)

1. `stage1_summary.json` mixes two references. The XGBoost, LightGBM and CatBoost entries were computed with reference 0.11946 (E060), for example XGBoost gap 0.12307 - 0.11946 = 0.0036. The `_rule` block and the `catboost_native` entry were rewritten later with reference 0.118066 (E061, an ensemble, registered in between): native gap 0.12212 - 0.11807 = 0.0041. The D19 decisions do not change with either reference (XGBoost would be 0.0050, still above 0.003), but the file is not internally consistent.
2. `stage2_rule` with 10 or fewer completed trials compares the first and last of the running best. For `catboost_native` (5 trials) this gives `improving_last_10 = true` and `stage2_recommended = true`, which is not meaningful.
3. EXPERIMENT_HISTORY states the gap rule as "< 0.003"; the code uses `<=`. Immaterial.
4. The 5 to 9 min per native trial in the checkpoint: measured 1.7 to 10.5 min, mean 5.6.
5. The registry-wide `reference` makes a base booster compete against the ensemble E061. This is conservative (the gap looks larger), not wrong.

---

## 8. Stage 2 results (to be filled by the director)

Stage 2 continues the CatBoost and LightGBM studies from 25 to 75 trials (D19, cap 60 min, command in the checkpoint section 4). XGBoost did not get Stage 2.

- CatBoost best after 75 trials: <<STAGE2: CatBoost best CV RMSE, trial number, parameters, registry id if registered>>
- LightGBM best after 75 trials: <<STAGE2: LightGBM best CV RMSE, trial number, parameters, registry id if registered>>
- Did either beat its Stage 1 score by more than 0.003? <<STAGE2: yes/no per booster, with the difference>>. The expected answer under hypothesis H1 is "no for CatBoost": E060 has a spread of 0.0028 between trials and was still dropping only slowly (0.12052 at trial 5, 0.11975 at 15, 0.11946 at 25).
- Minutes used: <<STAGE2: wall minutes for CatBoost, for LightGBM, and the new total against the 3 h cap (38.8 min before Stage 2)>>
- Reading to write after the numbers arrive: <<STAGE2: did TPE move toward the search-space edges noted in section 5.3, and is the improvement inside the 0.003 optimism band?>>

---

## 9. Questions you should be able to answer

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
