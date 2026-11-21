"""End-to-end pipeline: benchmark the RUL models, then compare maintenance policies on a simulated fleet.

    python -m src.run            # everything, ~3 minutes on a laptop
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lifelines import WeibullFitter

from .data import load_test, load_train, split_units
from .features import RegimeNormalizer, build_features, feature_columns
from .models import FleetLifeBaseline, RULModel, nasa_score, ridge_model, rmse
from .policies import PredictedRUL, RiskBased, RunToFailure, TimeBased
from .simulate import EngineTrack, SimConfig, run_many

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def featurize(norm: RegimeNormalizer, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    return build_features(norm.transform(df), norm.sensors_, f["windows"], f["slope_window"])


def with_target(feats: pd.DataFrame, raw: pd.DataFrame, cap: int) -> pd.DataFrame:
    rul = raw.sort_values(["unit", "cycle"])["rul"].to_numpy()
    return feats.assign(rul=np.minimum(rul, cap))


# --------------------------------------------------------------------------- benchmark

def benchmark(cfg: dict) -> dict:
    """NASA's official protocol: train on all training engines, score at the last cycle of each test engine."""
    raw_dir, cap = ROOT / cfg["data"]["raw_dir"], cfg["features"]["rul_cap"]
    results = {}
    for name in cfg["data"]["datasets"]:
        train = load_train(raw_dir, name)
        test, true = load_test(raw_dir, name)
        norm = RegimeNormalizer(cfg["features"]["n_regimes"]).fit(train)
        tr = with_target(featurize(norm, train, cfg), train, cap)
        te = featurize(norm, test, cfg).groupby("unit").tail(1).set_index("unit").loc[true.index]
        cols = feature_columns(tr)
        y_true = np.minimum(true.to_numpy(), cap)

        gbm = RULModel(cfg["model"]["lgbm"], [0.5], cfg["model"]["interval_coverage"]).fit(tr[cols], tr["rul"])
        ridge = ridge_model().fit(tr[cols], tr["rul"])
        fleet = FleetLifeBaseline(cap).fit(train["cycle"].to_numpy(), train["rul"].to_numpy())
        preds = {
            "Fleet-average life": fleet.predict(te["cycle"].to_numpy()),
            "Ridge regression": np.clip(ridge.predict(te[cols]), 0, cap),
            "LightGBM": np.clip(gbm.predict(te[cols]), 0, cap),
        }
        results[name] = {
            "test_engines": int(len(true)),
            "test_engines_capped": int((true > cap).sum()),
            "holdout_random_cut_rmse": holdout_check(train, cfg),
            "models": {k: {"rmse": round(rmse(p, y_true), 2), "nasa_score": round(nasa_score(p, y_true), 1)}
                       for k, p in preds.items()},
        }
        print(f"  {name}: " + ", ".join(f"{k} RMSE {v['rmse']}" for k, v in results[name]["models"].items()))
    return results


def holdout_check(train: pd.DataFrame, cfg: dict, seed: int = 3) -> float:
    """A second opinion on the official score: hold out 20% of the training engines, cut each
    at a random point in its life (30-100%), and score there. Harder than NASA's test set, which
    has many engines cut off early where the capped truth is simply 125."""
    cap = cfg["features"]["rul_cap"]
    g = split_units(train["unit"].unique(), {"fit": 0.8, "hold": 0.2}, seed)
    norm = RegimeNormalizer(cfg["features"]["n_regimes"]).fit(train[train["unit"].isin(g["fit"])])
    f = with_target(featurize(norm, train, cfg), train, cap)
    cols = feature_columns(f)
    fit = f[f["unit"].isin(g["fit"])]
    model = RULModel(cfg["model"]["lgbm"], [0.5], 0.8).fit(fit[cols], fit["rul"])
    rng = np.random.default_rng(seed)
    rows = [grp.iloc[rng.integers(int(0.3 * len(grp)), len(grp))] for _, grp in f[f["unit"].isin(g["hold"])].groupby("unit")]
    picks = pd.DataFrame(rows)
    return round(rmse(np.clip(model.predict(picks[cols]), 0, cap), picks["rul"]), 2)


# --------------------------------------------------------------------------- fleet simulation

def build_tracks(model: RULModel, feats: pd.DataFrame, cols: list[str], units) -> dict[int, EngineTrack]:
    tracks = {}
    sub = feats[feats["unit"].isin(units)]
    lo, med, hi = model.predict_band(sub[cols])
    point = model.predict(sub[cols])
    sub = sub.assign(point=point, lo=lo, med=med, hi=hi)
    for u, g in sub.groupby("unit"):
        tracks[int(u)] = EngineTrack(len(g), g["point"].to_numpy(), g["lo"].to_numpy(),
                                     g["med"].to_numpy(), g["hi"].to_numpy())
    return tracks


def weibull_age_replacement(lifetimes: np.ndarray, c_planned: float, c_failure: float) -> dict:
    """Classic MTBF-style analysis: fit a Weibull to engine lifetimes, then pick the replacement age T
    that minimises long-run cost per cycle = (Cp R(T) + Cf F(T)) / integral_0^T R(t) dt."""
    wf = WeibullFitter().fit(lifetimes)
    lam, rho = wf.lambda_, wf.rho_
    grid = np.arange(20, int(lifetimes.max()) + 1)
    surv = np.exp(-(grid / lam) ** rho)
    t = np.arange(1, int(lifetimes.max()) + 1)
    cum = np.cumsum(np.exp(-(t / lam) ** rho))
    rate = (c_planned * surv + c_failure * (1 - surv)) / cum[grid - 1]
    best = int(grid[np.argmin(rate)])
    return {"scale": float(lam), "shape": float(rho), "mtbf": float(np.mean(lifetimes)), "optimal_age": best}


def tune(policy_cls, grid, tracks, pool, capacity, sim, seeds):
    scored = [(run_many(tracks, pool, policy_cls(param=p), capacity, sim, seeds)["cost_per_1k"], p) for p in grid]
    return min(scored)[1]


def fleet_study(cfg: dict) -> dict:
    raw_dir, cap, name = ROOT / cfg["data"]["raw_dir"], cfg["features"]["rul_cap"], cfg["data"]["simulation_dataset"]
    raw = load_train(raw_dir, name)
    s = cfg["split"]
    groups = split_units(raw["unit"].unique(),
                         {k: s[k] for k in ("train", "calibrate", "tune", "test")}, s["seed"])

    train_raw = raw[raw["unit"].isin(groups["train"])]
    norm = RegimeNormalizer(cfg["features"]["n_regimes"]).fit(train_raw)
    feats = with_target(featurize(norm, raw, cfg), raw, cap)
    cols = feature_columns(feats)

    tr = feats[feats["unit"].isin(groups["train"])]
    cal = feats[feats["unit"].isin(groups["calibrate"])]
    model = RULModel(cfg["model"]["lgbm"], cfg["model"]["quantiles"], cfg["model"]["interval_coverage"])
    model.fit(tr[cols], tr["rul"]).calibrate(cal[cols], cal["rul"].to_numpy())

    # Did the calibrated band hold up on engines it never saw?
    held = feats[feats["unit"].isin(groups["test"])]
    lo, _, hi = model.predict_band(held[cols])
    y = held["rul"].to_numpy()
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    near = y <= 50
    coverage_near = float(np.mean((y[near] >= lo[near]) & (y[near] <= hi[near])))

    lifetimes = raw[raw["unit"].isin(groups["train"])].groupby("unit")["cycle"].max().to_numpy()
    sc = cfg["simulation"]
    weibull = weibull_age_replacement(lifetimes, sc["scheduled_cost"], sc["unscheduled_cost"])

    sim = SimConfig(**{k: sc[k] for k in SimConfig.__dataclass_fields__})
    tune_tracks = build_tracks(model, feats, cols, groups["tune"])
    test_tracks = build_tracks(model, feats, cols, groups["test"])
    tune_pool, test_pool = list(tune_tracks), list(test_tracks)
    tune_seeds = [s + 1000 for s in sc["seeds"][:5]]

    grids = {
        TimeBased: list(range(60, 321, 10)),
        PredictedRUL: list(range(0, 121, 4)),
        RiskBased: list(range(0, 121, 4)),
    }
    table = []
    for capacity in sc["capacities"]:
        rows = {"capacity": capacity, "policies": {}}
        policies = [RunToFailure()]
        for cls, grid in grids.items():
            policies.append(cls(param=tune(cls, grid, tune_tracks, tune_pool, capacity, sim, tune_seeds)))
        for pol in policies:
            m = run_many(test_tracks, test_pool, pol, capacity, sim, sc["seeds"])
            m["param"] = pol.param
            rows["policies"][pol.name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
        table.append(rows)
        best = min(rows["policies"].items(), key=lambda kv: kv[1]["cost_per_1k"])
        print(f"  capacity {capacity}: best {best[0]} cost/1k {best[1]['cost_per_1k']:.2f}")

    sensitivity = []
    for ratio in sc.get("sensitivity_ratios", []):
        sim_r = SimConfig(**{**{k: sc[k] for k in SimConfig.__dataclass_fields__}, "unscheduled_cost": ratio * sc["scheduled_cost"]})
        cap_r = sc["sensitivity_capacity"]
        row = {"ratio": ratio, "capacity": cap_r, "policies": {}}
        for cls, grid in grids.items():
            pol = cls(param=tune(cls, grid, tune_tracks, tune_pool, cap_r, sim_r, tune_seeds))
            m = run_many(test_tracks, test_pool, pol, cap_r, sim_r, sc["seeds"])
            row["policies"][pol.name] = {"param": pol.param, "cost_per_1k": round(m["cost_per_1k"], 3),
                                         "failures_per_1k": round(m["failures_per_1k"], 3),
                                         "availability": round(m["availability"], 4)}
        sensitivity.append(row)
        print(f"  cost ratio {ratio}: " + ", ".join(f"{k} {v['cost_per_1k']:.2f}" for k, v in row["policies"].items()))

    return {
        "dataset": name,
        "sensitivity": sensitivity,
        "engines": {k: int(len(v)) for k, v in groups.items()},
        "band": {"target_coverage": cfg["model"]["interval_coverage"], "coverage_test": round(coverage, 3),
                 "coverage_last_50_cycles": round(coverage_near, 3), "conformal_margin": round(model.margin_, 2)},
        "weibull": {k: round(v, 2) if isinstance(v, float) else v for k, v in weibull.items()},
        "simulation": sc,
        "results": table,
        "_model": model, "_feats": feats, "_cols": cols, "_groups": groups,
    }


# --------------------------------------------------------------------------- figures

def figures(study: dict, bench: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"Run to failure": "#9a948a", "Time-based": "#1f3b5c", "Predicted RUL": "#c88a1e", "Risk-based": "#c8371e"}

    # 1. One held-out engine: prediction and band against the truth.
    model, feats, cols, groups = study["_model"], study["_feats"], study["_cols"], study["_groups"]
    # Show the held-out engine with the median error over its last 100 cycles: typical, not cherry-picked.
    test = feats[feats["unit"].isin(groups["test"])]
    errs = {u: rmse(model.predict(g[cols])[-100:], g["rul"].to_numpy()[-100:]) for u, g in test.groupby("unit")}
    unit = sorted(errs, key=errs.get)[len(errs) // 2]
    g = feats[feats["unit"] == unit]
    lo, med, hi = model.predict_band(g[cols])
    fig, ax = plt.subplots(figsize=(8, 4))
    point = model.predict(g[cols])
    ax.fill_between(g["cycle"], lo, point, color="#c8371e", alpha=0.12, linewidth=0)
    ax.plot(g["cycle"], point, color="#c8371e", lw=2, label="Predicted RUL")
    ax.plot(g["cycle"], lo, color="#c8371e", lw=1, ls=":", label="Pessimistic bound (~10th pct, used by risk policy)")
    ax.plot(g["cycle"], g["rul"], color="#262626", lw=1.5, ls="--", label="True RUL (capped at 125)")
    ax.set_xlabel("Flight cycle"); ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_title(f"Held-out engine {unit} ({study['dataset']}), typical error", loc="left")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "engine_trace.png", dpi=150); plt.close(fig)

    # 2. Cost and failures as maintenance capacity changes.
    caps = [r["capacity"] for r in study["results"]]
    labels = [str(c) if c < study["simulation"]["fleet_size"] else "no limit" for c in caps]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pol in colors:
        axes[0].plot(labels, [r["policies"][pol]["cost_per_1k"] for r in study["results"]], marker="o", color=colors[pol], label=pol)
        axes[1].plot(labels, [r["policies"][pol]["failures_per_1k"] for r in study["results"]], marker="o", color=colors[pol], label=pol)
    axes[0].set_title("Maintenance cost per 1,000 fleet-cycles", loc="left")
    axes[1].set_title("In-service failures per 1,000 fleet-cycles", loc="left")
    for ax in axes:
        ax.set_xlabel("Maintenance slots per check")
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "policy_vs_capacity.png", dpi=150); plt.close(fig)


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    print("Benchmark on NASA test sets")
    bench = benchmark(cfg)
    print("Fleet study")
    study = fleet_study(cfg)
    figures(study, bench)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(bench, indent=2))
    public = {k: v for k, v in study.items() if not k.startswith("_")}
    (REPORTS / "fleet_study.json").write_text(json.dumps(public, indent=2))
    docs = ROOT / "docs"
    if docs.exists():
        (docs / "fleet_study.json").write_text(json.dumps(public, indent=2))
        (docs / "engine_trace.png").write_bytes((REPORTS / "figures" / "engine_trace.png").read_bytes())
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
