"""Append-only experiment registry.

Every experiment is one JSON line in ``experiments/registry.jsonl`` carrying the fields
required by the research protocol (hypothesis, rationale, implementation, evaluation,
results, interpretation, decision) plus enough configuration to re-run it with
``python -m houseprices.evaluate --experiment <id>``.

Re-running needs a *builder*: the dotted path of a zero-argument-free function such as
``houseprices.models:build_ridge`` and the keyword arguments it was called with. The
builder must return a complete estimator pipeline accepting the cleaned DataFrame.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from houseprices import config


@dataclass
class Experiment:
    # identity
    id: str                                  # e.g. "E012"
    agent: str                               # who ran it (director, feature-researcher, ...)
    title: str
    # research protocol
    hypothesis: str
    rationale: str
    implementation: str                      # files / features / params changed
    evaluation: str                          # e.g. "KFold(5, shuffle, seed 42), RMSE on log1p"
    # reproducibility
    builder: str                             # "module:function"
    builder_kwargs: dict = field(default_factory=dict)
    feature_set: str = "raw_cleaned"
    train_row_filter: str = "keep_all"       # name of a filter in houseprices.validation (or module:function)
    cv_n_splits: int = config.CV_N_SPLITS
    cv_n_repeats: int = 1
    seed: int = config.SEED
    # results
    fold_scores: list[float] = field(default_factory=list)
    cv_mean: float | None = None
    cv_std: float | None = None
    fit_seconds: float | None = None
    # conclusions
    interpretation: str = ""
    decision: str = "pending"                # retain | reject | investigate | pending
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "builder": self.builder,
                "builder_kwargs": self.builder_kwargs,
                "feature_set": self.feature_set,
                "train_row_filter": self.train_row_filter,
                "cv_n_splits": self.cv_n_splits,
                "cv_n_repeats": self.cv_n_repeats,
                "seed": self.seed,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_record(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.config_hash
        return d


def _registry_path() -> "config.Path":
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.REGISTRY_PATH


def load_registry() -> pd.DataFrame:
    """All experiments as a DataFrame (empty frame if none yet)."""
    path = _registry_path()
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def next_id() -> str:
    reg = load_registry()
    n = 0 if reg.empty else max(int(i[1:]) for i in reg["id"])
    return f"E{n + 1:03d}"


def get_experiment(experiment_id: str) -> dict:
    reg = load_registry()
    hit = reg.loc[reg["id"] == experiment_id] if not reg.empty else reg
    if hit.empty:
        raise KeyError(f"experiment {experiment_id} not found in {config.REGISTRY_PATH}")
    return hit.iloc[-1].to_dict()


def append(experiment: Experiment) -> Experiment:
    """Write one JSON line. Ids are assigned here if the experiment id is empty."""
    if not experiment.id:
        experiment.id = next_id()
    path = _registry_path()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(experiment.to_record(), default=str) + "\n")
    write_summary()
    return experiment


def write_summary() -> pd.DataFrame:
    """Regenerate experiments/registry_summary.csv, sorted by CV mean (best first)."""
    reg = load_registry()
    if reg.empty:
        return reg
    cols = [
        "id", "timestamp", "agent", "title", "feature_set", "train_row_filter",
        "cv_n_splits", "cv_n_repeats", "cv_mean", "cv_std", "fit_seconds", "decision", "config_hash",
    ]
    summary = reg[[c for c in cols if c in reg.columns]].sort_values("cv_mean", na_position="last")
    summary.to_csv(config.EXPERIMENTS_DIR / "registry_summary.csv", index=False)
    return summary
