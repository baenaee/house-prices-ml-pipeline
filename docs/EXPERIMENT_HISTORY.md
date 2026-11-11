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

## 2026-11-11 — Phase A (repair and reproduce), usage-limit checkpoint

- Agent A ported the 17 rules to `src/houseprices/cleaning.py` with the reconstructed C01 and the condition-based B04; `tests/test_cleaning.py` asserts every historical count. The repaired notebook `notebooks/01_cleaning_repaired.ipynb` executes top-to-bottom from `notebooks/` (80 cells, zero errors) and reproduces the C01 verification output (1/12 entries, 1/9 houses). Original notebook byte-identical (SHA-256 re-checked).
- Agent D round 1 found three real defects in the director's `validation.py` (F1 repeated-CV OOF overwritten, F2 NaN predictions silently scored, F3 empty training set after a filter not detected). All three were fixed by the director the same session; the strict xfail tests now pass as ordinary tests. Full suite: 77 passed.
- No experiment has been run yet. `experiments/registry.jsonl` does not exist; the first id will be E001.

| # | Decision | Alternatives considered | Ruling and evidence |
|---|---|---|---|
| D7 | pandas pinned to 2.3.3 in the research env | pandas 3.1.1 (resolved by pip) | pandas 3 changes the default string dtype; pinning keeps the repaired notebook's semantics identical to the original run. Code still uses `select_dtypes(include=["object","string"])` to be forward-compatible. |
| D8 | `cross_validate` raises on non-finite predictions and on an empty training fold | return NaN | A NaN score would be written to the registry and sorted last silently (Agent D, F2/F3). |

## Experiments

(appended as they run; format: id, hypothesis, rationale, implementation, evaluation, results, interpretation, decision)
