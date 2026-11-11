# Experiment history and decision log

Chronological. Each experiment also has a machine-readable record in `experiments/registry.jsonl` (summary in `experiments/registry_summary.csv`). This file adds the reasoning, the failures and the director's rulings on disagreements.

## 2026-11-11 — Phase 0: checkpoint, environment, skeleton

- Verified ZIP backup and Git checkpoint `fedba71` before any modification (see `docs/AUDIT_2026-11-11.md`).
- Created conda env `houseprices` (Python 3.12) with core packages only; TabPFN and GPU builds excluded by user decision.
- Skeleton: `src/houseprices/{config,validation,registry}.py`, `pyproject.toml`, `docs/TASK_BOARD.md`.

### Decisions

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D1 | Target is `log1p(SalePrice)`; metric is RMSE on that scale | raw-price RMSE; `log` instead of `log1p` | Competition metric is RMSLE; `log1p`/`expm1` is the exact inverse pair. |
| D2 | Selection protocol: shuffled 5-fold, seed 42; finalists RepeatedKFold 5x3 | single split; 10-fold | 1,460 rows: 5-fold keeps ~1,170 training rows per fold; repeats quantify variance for finalists. |
| D3 | Outlier removal only filters TRAINING rows of a fold | filter whole dataset before CV | Filtering validation rows would inflate scores by deleting hard cases (user correction 2). |
| D4 | 15 % holdout, stratified by price decile, single use after freezing | none; nested CV only | Cheap unbiased check of the whole selection procedure; usage logged (user correction 3). |
| D5 | B04 implemented as a condition (all basement fields missing), not `Id == 2121` | keep Id test | Generalisable rule; the condition selects exactly 2121 in test and nothing in train (asserted by test). |
| D6 | C01 reconstructed from documentation, labelled as reconstruction | claim recovery from backup | The cell is absent from every copy; only its documented behaviour and saved verification output exist. |

## Experiments

(appended as they run; format: id, hypothesis, rationale, implementation, evaluation, results, interpretation, decision)
