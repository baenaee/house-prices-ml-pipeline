DRAFT - status: complete. Interpretability and error analysis of the frozen final model E066 (single CatBoost, Optuna Stage 2, ruling D22).

# What the final model relies on and where it fails (E066)

Source: `experiments/interpretability/run_interpretability.py --experiment E066`. All numbers come from the tables in `outputs/interpretability/` (prefix `E066_`) unless stated. Errors are on the log1p(SalePrice) scale; residual = actual minus predicted, so a negative residual is an over-prediction. The out-of-fold (OOF) predictions use the selection folds (KFold 5, shuffle, seed 42) on the 1,241 development rows and reproduce the registered fold scores (CV mean 0.11807). The pooled OOF RMSE is 0.1190, slightly different from the mean of the fold RMSEs because folds differ in size. These are selection-fold numbers, so they are somewhat optimistic (about +0.002 to +0.003); the fresh-fold estimate for E066 is 0.1217.

## 1. What the model relies on

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

## 2. Where it fails

**By price** (`E066_residuals_by_price_decile.csv`, `outputs/figures/E066_residual_by_decile.png`, `E066_pred_vs_actual.png`, `E066_residual_vs_predicted.png`). The error is U-shaped: RMSE 0.215 in the cheapest decile (houses at or below 106,500) and 0.153 in the most expensive (278,000 and up), against 0.076 to 0.106 in deciles 2 to 9. The mean residuals show regression toward the mean: -0.097 for the cheapest decile (over-predicted) and +0.072 for the most expensive (under-predicted).

**By living area** (`E066_residuals_by_grlivarea.csv`). RMSE is 0.093 for 1,500 to 1,999 sq ft (393 houses), 0.134 below 1,000, 0.125 for 2,000 to 2,499, 0.201 for 3,000 to 3,999 (12 houses, mean residual +0.101) and 0.611 for 4,000 and above (4 houses).

**By neighbourhood** (`E066_residuals_by_neighborhood_top_bottom5.csv`). Worst RMSE: IDOTRR 0.239 (n=29), Edwards 0.165 (n=86), NoRidge 0.162 (n=34), StoneBr 0.157 (n=21), OldTown 0.156 (n=95). Best: CollgCr 0.069 (n=130), BrDale 0.066, Gilbert 0.063; NPkVill (n=7) and Blueste (n=2) are too small to read.

**The worst rows** (`E066_worst15_rows.csv`). The largest residual is Id 633 (-0.748, NWAmes, sold as a Family sale), then Ids 31, 463, 524 (-0.669), 1325, 1183 (+0.636) and 969. Four of the 15 are the houses above 4,000 sq ft (Ids 524, 692, 1183, 1299). Most of the others are cheap, old houses in IDOTRR, OldTown and Sawyer, or Partial and Abnormal sales, where the price probably reflects circumstances (condition of the sale, buyer) that the features capture only partly (my interpretation).

**How concentrated.** The worst 5 rows carry 13.6 percent of the total squared error and the worst 15 carry 31.1 percent, while only 6.6 percent of houses have an absolute residual above 0.2. Removing the four houses with GrLivArea above 4,000 lowers the pooled OOF RMSE from 0.1190 to 0.1140 (1,237 rows). Those four are predicted at 360,787 (actual 184,750, Id 524), 292,371 (160,000, Id 1299), 444,991 (755,000, Id 692) and 394,416 (745,000, Id 1183): the two Partial sales are over-predicted by a factor of about 1.8 to 1.95, the two NoRidge houses under-predicted by a factor of about 1.7 to 1.9.

## 3. Implications for the test set

Test Id 2550 is a 5,095 sq ft Edwards partial sale, the same profile as training Ids 524 and 1299, which sold for 184,750 and 160,000 and were over-predicted by 0.67 and 0.60. The submission predicts 268,438 for it. Two points follow. First, a tree model cannot extrapolate: every prediction is a sum of leaf values learned from training rows, and with only four training houses above 4,000 sq ft the model has almost no basis for a 5,095 sq ft house; 268,438 is a blend of the neighbours in the feature space, not an estimate of what such a house is worth. Second, whether the actual price resembles Ids 524 and 1299 (low, partial-sale effect) or Ids 692 and 1183 (very high) is exactly the question the training data cannot settle: the two pairs have opposite residual signs. One house cannot move the leaderboard score much, but expect this prediction to be wrong by a large factor in either direction. It should be treated as a known uncertainty, not tuned by hand.

## 4. Holdout residuals compared with the OOF pattern

The single holdout evaluation (219 houses, RMSE 0.10645) saved `experiments/finalists/holdout_predictions_E066.csv`; I only read the file. Splitting the holdout into terciles of actual price gives mean residual and RMSE of -0.023 and 0.114 (low tercile, 76 houses), +0.001 and 0.100 (middle, 71) and +0.035 and 0.104 (high, 72). The same terciles of the development OOF predictions give -0.035 and 0.144, +0.004 and 0.096, +0.030 and 0.113. The sign pattern is the same (cheap houses over-predicted, expensive under-predicted, nothing in the middle) with a smaller magnitude on the holdout. The holdout RMSE of 0.1065 is below the CV estimate of 0.1217, which fits a holdout with milder extreme errors (I did not inspect its areas) rather than a better model: its largest residuals are 0.41 and 0.40 (Ids 711 and 875), against 0.75 in development.

## 5. Limits of this analysis

- OOF residuals are from the selection folds and the model was selected on them; absolute levels are optimistic by 0.002 to 0.003.
- Permutation importance breaks the correlation structure when it shuffles a column, so it evaluates the model on unrealistic rows; it also shares credit among related columns.
- SHAP is in-sample and conditional on the model, and a sum over one-hot levels is a convention.
- Several groups are tiny (4 houses above 4,000 sq ft, 2 in Blueste, 7 in NPkVill); their RMSEs are anecdotes.
- Nothing here is causal. A column being important means the model uses it, not that changing it would change a price.
