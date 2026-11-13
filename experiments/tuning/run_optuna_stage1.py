"""C5: boosting Stage 1 - 25 Optuna TPE trials per booster (XGBoost, LightGBM, CatBoost).

    python experiments/tuning/run_optuna_stage1.py --feature-set base --train-row-filter keep_all
    python experiments/tuning/run_optuna_stage1.py --boosters catboost --catboost-native
    python experiments/tuning/run_optuna_stage1.py --dry-run          # plumbing check, registers nothing

Objective = mean 5-fold CV RMSE (log scale) from the shared protocol. The parameters of a
trial are exactly what ``models.OPTUNA_SPACES[name](trial)`` returns (not rebuilt from
``trial.params``, because the LightGBM space fixes ``subsample_freq`` outside the trial),
and those COMPLETE params are what gets registered, so ``houseprices.evaluate`` can rebuild
the estimator with ``resolve_callable(builder)(**builder_kwargs)``.

Studies are stored in ``experiments/tuning/optuna_stage1.db`` (git-ignored) and are
resumable: a re-run only adds the missing trials. A booster whose best config is already in
the registry is not registered twice.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "models"))
import _common  # noqa: E402
from houseprices import data, models, registry, validation  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

BUILDERS = {
    "xgboost": "houseprices.models:build_xgboost",
    "lightgbm": "houseprices.models:build_lightgbm",
    "catboost": "houseprices.models:build_catboost",
}
SIZE_KEY = {"xgboost": "n_estimators", "lightgbm": "n_estimators", "catboost": "iterations"}
STAGE2_GAP = 0.003          # director's rule: Stage 2 if within this of the best model ...
STAGE2_WINDOW = 10          # ... or still improving over the last 10 trials
IMPROVE_TOL = 1e-4          # "improving" = best-so-far dropped by more than this in the window
TUNING_CAP_HOURS = 3.0


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.write_text("", encoding="utf-8") if not path.exists() else None

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def make_objective(name: str, extra: dict, fs: str, flt: str, dry: bool):
    builder = BUILDERS[name]

    def objective(trial: optuna.Trial) -> float:
        params = models.OPTUNA_SPACES[name](trial)       # COMPLETE params, incl. fixed ones
        if dry:                                           # tiny models, plumbing only
            params = {**params, SIZE_KEY[name]: 20}
        kwargs = {**params, **extra}
        r = _common.run_cv(builder, kwargs, fs, flt)
        trial.set_user_attr("builder_kwargs", json.dumps(kwargs))
        trial.set_user_attr("fold_scores", json.dumps([round(s, 6) for s in r.fold_scores]))
        trial.set_user_attr("cv_std", r.std)
        trial.set_user_attr("seconds", round(r.fit_seconds, 2))
        return r.mean

    return objective


def run_study(name: str, label: str, extra: dict, args, log: Logger, storage, soft_deadline: float,
              out_dir: Path) -> tuple[optuna.Study, float, bool]:
    study = optuna.create_study(
        study_name=f"stage1_{label}_{args.feature_set}_{args.train_row_filter}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage, load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    todo = max(0, args.n_trials - done)
    t0 = time.perf_counter()
    capped = False
    if done:
        log(f"{label}: resuming study with {done} completed trials; {todo} to run")

    def callback(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        nonlocal capped
        n = len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n % 5 == 0 or n == args.n_trials:
            log(f"{label}: trial {tr.number} done ({n} complete), last={tr.value:.5f}, "
                f"best={st.best_value:.5f}, elapsed={(time.perf_counter() - t0) / 60:.1f} min")
        if time.time() > soft_deadline and n < args.n_trials:
            capped = True
            log(f"{label}: time cap reached after {n} trials; stopping this study")
            st.stop()

    if todo:
        study.optimize(make_objective(name, extra, args.feature_set, args.train_row_filter, args.dry_run),
                       n_trials=todo, callbacks=[callback])
    return study, time.perf_counter() - t0, capped


def trials_table(study: optuna.Study) -> pd.DataFrame:
    df = study.trials_dataframe()
    return df[df["state"] == "COMPLETE"].reset_index(drop=True)


def stage2_rule(values: list[float], reference: float) -> dict:
    best = float(np.min(values))
    running = np.minimum.accumulate(values)
    if len(values) > STAGE2_WINDOW:
        improving = bool(running[-STAGE2_WINDOW - 1] - running[-1] > IMPROVE_TOL)
    else:
        improving = bool(running[0] - running[-1] > IMPROVE_TOL)
    gap = best - reference
    return dict(best_value=best, gap_to_best_model=gap, within_gap=bool(gap <= STAGE2_GAP),
                improving_last_10=improving, stage2_recommended=bool(gap <= STAGE2_GAP or improving))


def register_best(name: str, label: str, study: optuna.Study, elapsed: float, capped: bool, args) -> None:
    builder = BUILDERS[name]
    complete = trials_table(study)
    best = study.best_trial
    kwargs = json.loads(best.user_attrs["builder_kwargs"])
    # the registered kwargs must rebuild exactly the estimator the objective evaluated
    assert json.loads(json.dumps(kwargs)) == kwargs, "builder_kwargs are not JSON round-trippable"
    est = data.resolve_callable(builder)(**kwargs)
    again = data.resolve_callable(builder)(**json.loads(json.dumps(kwargs)))
    assert repr(est) == repr(again), "rebuilt pipeline differs"
    if name == "lightgbm":
        assert kwargs["subsample_freq"] == 1

    fs, flt = args.feature_set, args.train_row_filter
    eid = _common.find_registered(builder, kwargs, fs, flt)
    if eid:
        print(f"[skip] best {label} config already registered as {eid}")
        return
    top3 = complete.nsmallest(3, "value")
    top_txt = "; ".join(
        f"#{int(r['number'])}: {r['value']:.5f} {json.loads(r['user_attrs_builder_kwargs'])}" for _, r in top3.iterrows())
    fold_scores = json.loads(best.user_attrs["fold_scores"])
    result = validation.CVResult(fold_scores, np.array([]), float(best.user_attrs["seconds"]))
    n_done = len(complete)
    reg = registry.load_registry()
    ref_txt = ""
    if name == "xgboost" and not reg.empty:
        hit = reg[reg["title"].str.contains("XGBoost defaults", na=False)]
        if not hit.empty:
            d = float(hit.iloc[-1]["cv_mean"])
            ref_txt = f" Untuned XGBoost scored {d:.4f}; tuning changed it by {best.value - d:+.4f}."
    cap_txt = " The shared time cap stopped the study early." if capped else ""
    interp = (f"Best of {n_done} TPE trials: RMSE {best.value:.4f} (fold std {best.user_attrs['cv_std']:.4f}).{ref_txt} "
              f"Median trial {complete['value'].median():.4f}, worst {complete['value'].max():.4f}, so the search space "
              f"{'is sensitive to the settings' if complete['value'].max() - best.value > 0.01 else 'is fairly forgiving'}.{cap_txt}")
    _common.run_and_register(
        exp_id="", title=f"Optuna stage 1: {label} best of {n_done} trials",
        hypothesis=f"A bounded TPE search ({n_done} trials) finds {label} settings clearly better than library defaults.",
        rationale="Boosters are sensitive to learning rate, depth and regularisation; a small seeded TPE search gives a "
                  "fair, budget-bounded tuned score per booster before deciding on a larger Stage 2.",
        implementation=f"{builder}({kwargs}); Optuna TPESampler(seed=42), objective = mean 5-fold CV RMSE; "
                       f"space models.OPTUNA_SPACES['{name}']; study stage1_{label}_{fs}_{flt}.",
        builder=builder, builder_kwargs=kwargs, feature_set=fs, train_row_filter=flt,
        interpret=lambda r, t=interp: (t, "retain"), result=result,
        tags=["tuning", "stage1", name] + (["native-categoricals"] if label.endswith("native") else []),
        notes=f"n_trials={n_done}; elapsed={elapsed / 60:.1f} min; top-3 trials: {top_txt}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="base")
    ap.add_argument("--train-row-filter", default="keep_all")
    ap.add_argument("--boosters", nargs="+", default=["xgboost", "lightgbm", "catboost"], choices=list(BUILDERS))
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--catboost-native", action="store_true", help="CatBoost with native_categoricals=True")
    ap.add_argument("--time-cap-minutes", type=float, default=45.0, help="shared cap for the whole stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="2 trials, tiny models, first 150 rows, outputs in a temp dir, registers nothing")
    args = ap.parse_args()
    if args.dry_run:
        args.n_trials = 2
        _common.set_dry_run(150)
        out_dir = Path(tempfile.mkdtemp(prefix="stage1_dry_"))
        storage = None                       # in-memory
    else:
        out_dir = HERE
        storage = f"sqlite:///{(HERE / 'optuna_stage1.db').as_posix()}"
    fs = args.feature_set

    start = time.time()
    cap_s = args.time_cap_minutes * 60
    summary_path = out_dir / "stage1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    finished: dict[str, tuple[optuna.Study, float, bool, str]] = {}

    for i, name in enumerate(args.boosters):
        label = "catboost_native" if (name == "catboost" and args.catboost_native) else name
        extra = {"native_categoricals": True} if label == "catboost_native" else {}
        remaining = cap_s - (time.time() - start)
        share = remaining / (len(args.boosters) - i)           # fair share of what is left
        log = Logger(out_dir / f"stage1_{label}_{fs}.log")
        log(f"{label}: start, {args.n_trials} trials, soft budget {share / 60:.1f} min, "
            f"filter={args.train_row_filter}, threads={models.N_THREADS}")
        study, elapsed, capped = run_study(name, label, extra, args, log, storage, time.time() + share, out_dir)
        if not study.trials:
            continue
        trials_table(study).to_csv(out_dir / f"stage1_{label}_{fs}_trials.csv", index=False)
        finished[label] = (study, elapsed, capped, name)
        log(f"{label}: finished, best={study.best_value:.5f}, elapsed={elapsed / 60:.1f} min")

    # Stage-2 rule: reference = best CV mean among registered experiments and these studies.
    reg = registry.load_registry()
    ref_candidates = [s.best_value for s, *_ in finished.values()]
    if not reg.empty and reg["cv_mean"].notna().any():
        ref_candidates.append(float(reg["cv_mean"].min()))
    reference = float(min(ref_candidates)) if ref_candidates else float("nan")
    for label, (study, elapsed, capped, name) in finished.items():
        values = [t.value for t in sorted(study.trials, key=lambda t: t.number)
                  if t.state == optuna.trial.TrialState.COMPLETE]
        rule = stage2_rule(values, reference)
        summary[label] = dict(
            **rule, n_trials=len(values), elapsed_minutes=round(elapsed / 60, 2), stopped_by_time_cap=capped,
            best_params=json.loads(study.best_trial.user_attrs["builder_kwargs"]),
            feature_set=fs, train_row_filter=args.train_row_filter)
    summary["_rule"] = dict(
        reference_best_model_cv=reference, gap_threshold=STAGE2_GAP, improving_window=STAGE2_WINDOW,
        improving_tolerance=IMPROVE_TOL, overall_tuning_cap_hours=TUNING_CAP_HOURS,
        note="reference = lowest cv_mean in the registry (any model) or this stage's best, whichever is lower; "
             "improving = best-so-far dropped by more than the tolerance across the last 10 trials",
        stage_elapsed_minutes=round((time.time() - start) / 60, 2))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary written to {summary_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "_rule"}, indent=2, default=str)[:3000])

    if args.dry_run:
        for label, (study, _, _, name) in finished.items():
            kwargs = json.loads(study.best_trial.user_attrs["builder_kwargs"])
            est = data.resolve_callable(BUILDERS[name])(**kwargs)
            print(f"[dry-run] {label}: rebuild OK -> {type(est).__name__}; nothing registered")
        return
    for label, (study, elapsed, capped, name) in finished.items():
        register_best(name, label, study, elapsed, capped, args)


if __name__ == "__main__":
    main()
