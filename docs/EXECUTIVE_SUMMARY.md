# Executive summary: House Prices (Kaggle, Ames)

**What this is.** A complete, reproducible machine-learning project that predicts house sale prices for the Kaggle competition *House Prices: Advanced Regression Techniques*, built on top of an existing data-cleaning notebook. It is designed to show sound ML engineering practice: leakage-safe validation, logged experiments, an independent audit, and honest reporting of uncertainty.

## Result

| | RMSLE (lower is better) |
|---|---|
| Baseline: predict the average price | 0.399 |
| Baseline: linear regression on numeric columns | 0.160 |
| Best linear model (Lasso), on folds not used for tuning | 0.128 |
| **Final model (tuned CatBoost), on folds not used for tuning** | **0.122** |
| Final model on a locked 219-house holdout, scored once | 0.106 |

The expected error of the final model on new houses is about **0.12** on the competition metric, which on the holdout corresponded to a mean absolute price error of about 8 % and a median of 5 %. The holdout came out better than that; with 219 houses, and none of the four hardest houses among them, it is treated as a consistency check and not as the headline.

The submission file is produced and validated. It has deliberately not been submitted: the public leaderboard would be one more score to overfit.

## How it was done

- **83 logged experiments** in an append-only registry: baselines, feature ablations, outlier policies, 136 linear grid points, 180 boosting trials in a staged Optuna search, 12 ensembles, and a finalist round. Any registered score can be reproduced with one command.
- **No leakage by construction.** Every step that learns from data (imputation, skew correction, encoding, scaling, model, ensemble weights) is fitted inside each cross-validation training fold through scikit-learn pipelines.
- **Measured selection bias.** Scores on the folds used for choosing were 0.003–0.004 too good; all candidates were re-scored on fresh folds and those numbers are the ones reported.
- **A single-use holdout enforced in code.** One evaluation, after the model was frozen in Git and approved by the auditor, with the acceptable range written down beforehand.
- **Independent audit.** A separate reviewer read all pipeline code over seven rounds plus a final round and logged 36 findings, including real bugs in the validation code; each code bug has a test, while informational notes are documented without one.

## Findings worth knowing

1. **Removing the "famous outliers" hurt.** When removal is applied only to training data, as it must be, every model got worse. The gain usually reported comes from also deleting those houses from the validation data.
2. **Ensembling did not pay.** The best ensemble (a plain average of Lasso and CatBoost) was 0.001 better than CatBoost alone, one third of a standard error. Learned blend weights and stacking never beat the plain average. The simpler single model was chosen, by a rule written before the comparison was read.
3. **Tuning has a plateau.** Going from 25 to 75 tuning trials improved CatBoost by 0.0016; 58 of the 75 trials were within 0.003 of the best.
4. **The error lives in a few houses.** Fifteen of 1,241 houses carry 31 % of the squared error; cheap houses are over-predicted and expensive ones under-predicted.

## Limitations

- The final choice between CatBoost alone and the Lasso + CatBoost average is within statistical noise.
- All selection used the same 1,241 houses; the only untouched data is a 219-house holdout (95 % interval of its score: 0.092–0.121).
- Very large or atypical houses are predicted poorly, and the test set contains at least one.
- One city, 2006–2010; no claim about other markets or periods.

## Where to look

`README.md` (overview and commands) · `docs/RESEARCH_REPORT.md` (full results) · `docs/STUDY_GUIDE.md` (methods explained) · `docs/EXPERIMENT_HISTORY.md` (every decision with its evidence) · `docs/VALIDATION_AUDIT.md` (the audit) · `docs/CHATGPT_AUDIT_PACKET.md` (claims and the commands that verify them).

*Built by a research director model coordinating four specialist agents (features, modelling, ensembles, validation audit) under the owner's methodological rules; the owner reviewed the plan and set the binding constraints.*
