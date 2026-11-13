"""Shared plumbing for the feature-ablation and LotFrontage scripts (Agent A).

* Models are registry builders ("module:function" + kwargs) so ``python -m houseprices.evaluate``
  can reproduce every experiment.
* A config that already exists in the registry (same config_hash) is reused, never re-run.
* Interpretations quote the paired difference against a reference run on the SAME folds:
  mean of per-fold differences and its standard error (std of differences / sqrt(k)).
* ``--dry-run`` never calls cross-validation and never writes the registry: it fits each
  configuration once on the first rows of the development data and prints a holdout-free RMSE.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import _common  # noqa: E402

from houseprices import data, registry  # noqa: E402

AGENT = "feature-researcher"
DRY_RUN_ROWS = 150

MODELS: dict[str, tuple[str, dict]] = {
    "ridge": ("houseprices.models:build_ridge", {"alpha": 10.0}),
    "elasticnet": ("houseprices.models:build_elasticnet", {"alpha": 0.001, "l1_ratio": 0.5}),
    "xgboost": ("houseprices.models:build_xgboost", {}),
}
LINEAR = ("ridge", "elasticnet")


def model_spec(model: str, preprocessor_kwargs: dict | None = None) -> tuple[str, dict]:
    builder, kwargs = MODELS[model]
    kwargs = dict(kwargs)
    if preprocessor_kwargs:
        kwargs["preprocessor_kwargs"] = dict(preprocessor_kwargs)
    return builder, kwargs


def paired_stats(scores: list[float], ref: list[float]) -> tuple[float, float]:
    """(mean difference scores-ref, standard error of the per-fold differences). Negative = better."""
    d = np.asarray(scores) - np.asarray(ref)
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se


def compare_text(label: str, scores: list[float], ref: list[float], ref_name: str) -> tuple[str, str]:
    """Return (sentence, verdict) with verdict in {'better', 'worse', 'tie'} (beyond 1 paired SE)."""
    delta, se = paired_stats(scores, ref)
    if abs(delta) > se:
        verdict = "better" if delta < 0 else "worse"
        beyond = "exceeds"
    else:
        verdict, beyond = "tie", "does not exceed"
    sentence = (f"{label}: RMSE {np.mean(scores):.4f} vs {np.mean(ref):.4f} for {ref_name}; "
                f"paired difference {delta:+.4f} (SE {se:.4f}), which {beyond} one paired standard error "
                f"({'lower RMSE = better' if delta < 0 else 'higher RMSE = worse'}).")
    return sentence, verdict


class Runner:
    """Runs (or reuses) one experiment per configuration and remembers the fold scores."""

    def __init__(self, dry_run: bool, budget_minutes: float):
        self.dry_run = dry_run
        self.deadline = time.time() + budget_minutes * 60
        self.scores: dict[tuple, dict] = {}   # key -> {"folds": [...], "mean": float, "id": str}
        self.skipped_for_budget: list[tuple] = []
        if dry_run:
            _common.set_dry_run(DRY_RUN_ROWS)

    def _dry_fit(self, builder: str, kwargs: dict, feature_set: str) -> float:
        dev = _common.get_data(feature_set)
        n = len(dev.X)
        cut = int(n * 0.8)
        est = data.resolve_callable(builder)(**kwargs)
        est.fit(dev.X.iloc[:cut], dev.y_log.iloc[:cut])
        pred = est.predict(dev.X.iloc[cut:])
        return float(np.sqrt(np.mean((pred - dev.y_log.iloc[cut:].to_numpy()) ** 2)))

    def get(self, key: tuple, feature_set: str, builder: str, kwargs: dict, *, title: str, hypothesis: str,
            rationale: str, implementation: str, interpret, tags: list[str]) -> dict | None:
        """Return {"folds", "mean", "id"} for this config, running/registering it if needed.

        ``interpret(result) -> (interpretation, decision)`` receives the CVResult.
        """
        if self.dry_run:
            rmse = self._dry_fit(builder, kwargs, feature_set)
            print(f"  [dry-run] {key}: fit/predict OK, holdout-free RMSE on last 20% of {DRY_RUN_ROWS} rows = {rmse:.4f}")
            self.scores[key] = {"folds": [rmse] * 5, "mean": rmse, "id": "dry"}
            return self.scores[key]
        existing = _common.find_registered(builder, kwargs, feature_set)
        if existing:
            rec = registry.get_experiment(existing)
            print(f"  skip {key}: already registered as {existing}")
            self.scores[key] = {"folds": list(rec["fold_scores"]), "mean": float(rec["cv_mean"]), "id": existing}
            return self.scores[key]
        if time.time() > self.deadline:
            self.skipped_for_budget.append(key)
            print(f"  BUDGET EXHAUSTED, not run: {key}")
            return None
        t0 = time.time()
        res = _common.run_cv(builder, kwargs, feature_set)
        exp = _common.run_and_register(
            "", title, hypothesis, rationale, implementation, builder, kwargs, interpret,
            feature_set=feature_set, tags=tags, result=res, agent=AGENT)
        print(f"  ran {key}: {exp.id} cv_mean={res.mean:.4f} ({time.time() - t0:.0f}s) -> {exp.decision}")
        self.scores[key] = {"folds": list(res.fold_scores), "mean": res.mean, "id": exp.id}
        return self.scores[key]
