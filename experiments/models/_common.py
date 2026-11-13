"""Shared helper for the model experiment scripts: run one CV and register it.

Keeps the run scripts short and guarantees every experiment is evaluated the same way:
same data entry point, same seeded folds, same registry fields.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from houseprices import config, data, registry, validation

AGENT = "model-researcher"
EVALUATION = "KFold(5, shuffle, seed 42) on log1p(SalePrice); metric RMSE on log scale; preprocessing refit per fold"


_DRY_RUN_ROWS: int | None = None


def set_dry_run(n_rows: int | None) -> None:
    """Dry-run mode: ``get_data`` returns the first ``n_rows`` rows, with NO holdout split.

    Used only to validate script plumbing; it never creates ``holdout_ids.json``.
    """
    global _DRY_RUN_ROWS
    _DRY_RUN_ROWS = n_rows


@lru_cache(maxsize=4)
def _load(feature_set: str, with_holdout_split: bool) -> data.DevelopmentData:
    return data.load_development_data(feature_set, with_holdout_split=with_holdout_split)


def get_data(feature_set: str) -> data.DevelopmentData:
    """Load (and cache) the development data. The holdout inside is never used."""
    if _DRY_RUN_ROWS is None:
        return _load(feature_set, True)
    full = _load(feature_set, False)
    return data.DevelopmentData(full.X.head(_DRY_RUN_ROWS), full.y_log.head(_DRY_RUN_ROWS),
                                full.X_holdout, full.y_holdout_log, full.X_test, full.feature_groups)


def run_cv(builder: str, builder_kwargs: dict, feature_set: str = "base",
           train_row_filter: str = "keep_all", n_repeats: int = 1) -> validation.CVResult:
    """Cross-validate a registry-style builder without registering anything."""
    dev = get_data(feature_set)
    estimator = data.resolve_callable(builder)(**builder_kwargs)
    cv = validation.make_cv(config.CV_N_SPLITS, n_repeats, config.SEED)
    row_filter = data.get_train_row_filter(train_row_filter)
    return validation.cross_validate(estimator, dev.X, dev.y_log, cv, row_filter)


def run_and_register(
    exp_id: str,
    title: str,
    hypothesis: str,
    rationale: str,
    implementation: str,
    builder: str,
    builder_kwargs: dict,
    interpret: Callable[[validation.CVResult], tuple[str, str]],
    feature_set: str = "base",
    train_row_filter: str = "keep_all",
    n_repeats: int = 1,
    tags: list[str] | None = None,
    notes: str = "",
    result: validation.CVResult | None = None,
    agent: str = AGENT,
) -> registry.Experiment:
    """Run the CV (unless ``result`` is supplied) and append a fully filled Experiment.

    ``interpret(result) -> (interpretation, decision)`` is called AFTER scoring so the text
    can quote the actual numbers.
    """
    if _DRY_RUN_ROWS is not None:
        raise RuntimeError("dry-run mode is active: scores from truncated data must never be registered")
    existing = registry.load_registry()
    if exp_id and not existing.empty and exp_id in set(existing["id"]):
        raise ValueError(f"{exp_id} already exists in the registry; refusing to duplicate it")
    result = result or run_cv(builder, builder_kwargs, feature_set, train_row_filter, n_repeats)
    interpretation, decision = interpret(result)
    exp = registry.Experiment(
        id=exp_id,
        agent=agent,
        title=title,
        hypothesis=hypothesis,
        rationale=rationale,
        implementation=implementation,
        evaluation=EVALUATION if n_repeats == 1 else EVALUATION.replace("KFold(5,", f"RepeatedKFold(5x{n_repeats},"),
        builder=builder,
        builder_kwargs=builder_kwargs,
        feature_set=feature_set,
        train_row_filter=train_row_filter,
        cv_n_splits=config.CV_N_SPLITS,
        cv_n_repeats=n_repeats,
        fold_scores=[round(s, 6) for s in result.fold_scores],
        cv_mean=round(result.mean, 6),
        cv_std=round(result.std, 6),
        fit_seconds=round(result.fit_seconds, 2),
        interpretation=interpretation,
        decision=decision,
        tags=tags or [],
        notes=notes,
    )
    registry.append(exp)
    return exp


def find_registered(builder: str, builder_kwargs: dict, feature_set: str = "base",
                    train_row_filter: str = "keep_all", n_repeats: int = 1,
                    seed: int = config.SEED) -> str | None:
    """Return the id of an existing experiment with the same config_hash, else None.

    ``seed`` is part of the hash (audit F26): finalist checks use ``config.CV_SEED_FINALISTS``.
    """
    probe = registry.Experiment(
        id="", agent=AGENT, title="", hypothesis="", rationale="", implementation="", evaluation="",
        builder=builder, builder_kwargs=builder_kwargs, feature_set=feature_set,
        train_row_filter=train_row_filter, cv_n_splits=config.CV_N_SPLITS, cv_n_repeats=n_repeats,
        seed=seed,
    )
    reg = registry.load_registry()
    if reg.empty or "config_hash" not in reg.columns:
        return None
    hit = reg.loc[reg["config_hash"] == probe.config_hash]
    return None if hit.empty else str(hit.iloc[-1]["id"])
