"""Fit a registered configuration on all available rows and save the artefact.

    python -m houseprices.train --experiment E060                 # fit on development rows (1,241)
    python -m houseprices.train --experiment E060 --rows all      # fit on all 1,460 training rows
    python -m houseprices.train --experiment E060 --rows all --override '{"preprocessor_kwargs": {"clip_years": true}}'

This command produces the fitted model only. It is NOT evidence of cross-validated
performance: that evidence comes exclusively from ``python -m houseprices.evaluate``.

Rows:
* ``development`` (default): the 1,241 rows outside the single-use holdout. Use this for the
  model that is scored once on the holdout with ``validation.score_holdout_once``.
* ``all``: every training row including the holdout. Use this ONLY after the holdout has been
  scored and the procedure frozen, for the final submission model.

The training-row filter recorded in the registry is applied (ruling D15: keep_all).
An ``--override`` JSON is merged one level deep into ``builder_kwargs`` (e.g. to switch on the
fold-safe year clipper, which is a no-op on the training rows); it is written into the
artefact's metadata so nothing is silent.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from houseprices import config
from houseprices import data as data_mod
from houseprices import registry, validation


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - git may be absent from PATH
        return "unknown"


def _library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            versions[name] = "not installed"
    return versions


def merge_override(builder_kwargs: dict, override: dict | None) -> dict:
    """One-level-deep merge: nested dicts (e.g. preprocessor_kwargs) are updated, not replaced."""
    merged = json.loads(json.dumps(builder_kwargs))  # deep copy of JSON-able kwargs
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def fit_registered(experiment_id: str, rows: str = "development", override: dict | None = None,
                   out_dir: Path | None = None, verbose: bool = True) -> tuple[Path, dict]:
    """Fit the registered builder on ``rows`` and save ``<id>_<rows>.joblib`` plus a JSON sidecar."""
    if rows not in ("development", "all"):
        raise ValueError("rows must be 'development' or 'all'")
    rec = registry.get_experiment(experiment_id)
    kwargs = rec.get("builder_kwargs") or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    kwargs = merge_override(kwargs, override)
    estimator = data_mod.resolve_callable(rec["builder"])(**kwargs)

    feature_set = rec.get("feature_set", "base")
    dev = data_mod.load_development_data(feature_set, with_holdout_split=(rows == "development"))
    row_filter = data_mod.get_train_row_filter(rec.get("train_row_filter", "keep_all"))
    keep = np.asarray(row_filter(dev.X, dev.y_log), dtype=bool)
    X_fit, y_fit = dev.X.loc[keep], dev.y_log.loc[keep]

    t0 = time.perf_counter()
    estimator.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - t0

    out_dir = Path(out_dir) if out_dir is not None else config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{experiment_id}_{rows}.joblib"
    meta = {
        "experiment_id": experiment_id,
        "title": rec.get("title"),
        "config_hash": rec.get("config_hash"),
        "registered_cv_mean": rec.get("cv_mean"),
        "builder": rec["builder"],
        "builder_kwargs": kwargs,
        "override": override or {},
        "feature_set": feature_set,
        "train_row_filter": rec.get("train_row_filter", "keep_all"),
        "rows": rows,
        "n_rows_fit": int(len(X_fit)),
        "n_rows_available": int(len(dev.X)),
        "fit_seconds": round(fit_seconds, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
        "note": "Fitted artefact only. CV evidence: python -m houseprices.evaluate --experiment "
                + experiment_id,
    }
    joblib.dump(estimator, model_path)
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps({k: v for k, v in meta.items() if k != "library_versions"}, indent=2))
        print(f"saved {model_path}")
    return model_path, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="registry id, e.g. E060")
    parser.add_argument("--rows", choices=["development", "all"], default="development")
    parser.add_argument("--override", default=None, help="JSON merged into builder_kwargs (logged in the sidecar)")
    args = parser.parse_args(argv)
    override = json.loads(args.override) if args.override else None
    fit_registered(args.experiment, args.rows, override)
    return 0


if __name__ == "__main__":
    sys.exit(main())
