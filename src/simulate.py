"""Fleet simulation: maintenance decisions under a limited number of slots per check.

Each fleet position holds one engine. Engines fly until they are pulled for planned
maintenance (short downtime, small cost) or fail in service (long downtime, large cost).
Either way a fresh engine from the held-out pool replaces them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import Policy


@dataclass
class EngineTrack:
    """Predictions for one held-out run-to-failure engine, indexed by age (cycles flown)."""

    life: int                 # the engine fails at the end of this cycle
    point: np.ndarray         # predicted RUL after cycle a is point[a - 1]
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray


@dataclass
class SimConfig:
    fleet_size: int
    horizon: int
    check_interval: int
    scheduled_downtime: int
    unscheduled_downtime: int
    scheduled_cost: float
    unscheduled_cost: float
    initial_age_max: float


def draw_schedule(pool: list[int], tracks: dict[int, EngineTrack], cfg: SimConfig, seed: int, depth: int = 64):
    """Pre-draw, per fleet position, the sequence of engines it will receive and its starting age.
    Every policy sees the same draws (common random numbers), so differences come from decisions."""
    rng = np.random.default_rng(seed)
    queues = rng.choice(pool, size=(cfg.fleet_size, depth), replace=True)
    first_ages = np.array([
        int(rng.uniform(0, cfg.initial_age_max) * tracks[int(queues[i, 0])].life)
        for i in range(cfg.fleet_size)
    ])
    return queues, first_ages


def simulate(tracks: dict[int, EngineTrack], schedule, policy: Policy, capacity: int, cfg: SimConfig) -> dict:
    queues, first_ages = schedule
    n = cfg.fleet_size
    nxt = np.ones(n, dtype=int)                       # next index into each position's queue
    engine = queues[:, 0].astype(int).copy()
    age = first_ages.copy()
    down_until = np.full(n, -1)                       # >= t means the position is in maintenance; -1 = flying

    flown = 0
    planned = failures = 0
    wasted = []                                       # remaining life thrown away at planned removals
    t = 0
    while t < cfg.horizon:
        # 1. Decide which flying engines to pull at this check.
        candidates = []
        for i in range(n):
            if down_until[i] >= t or age[i] == 0:
                continue
            tr = tracks[engine[i]]
            k = age[i] - 1
            want, priority = policy.decide(age[i], tr.point[k], tr.lower[k], tr.median[k], tr.upper[k])
            if want:
                candidates.append((priority, i))
        candidates.sort()
        for _, i in candidates[:capacity]:
            planned += 1
            wasted.append(tracks[engine[i]].life - age[i])
            down_until[i] = t + cfg.scheduled_downtime

        # 2. Fly the interval.
        end = min(t + cfg.check_interval, cfg.horizon)
        for i in range(n):
            c = t
            while c < end:
                if down_until[i] >= c:
                    if down_until[i] >= end:
                        break
                    c = down_until[i]
                    engine[i] = int(queues[i, nxt[i] % queues.shape[1]])
                    nxt[i] += 1
                    age[i] = 0
                    down_until[i] = -1
                tr = tracks[engine[i]]
                fly = min(end - c, tr.life - age[i])
                age[i] += fly
                flown += fly
                c += fly
                if age[i] >= tr.life:                 # in-service failure
                    failures += 1
                    down_until[i] = c + cfg.unscheduled_downtime
        t = end

    total = n * cfg.horizon
    cost = planned * cfg.scheduled_cost + failures * cfg.unscheduled_cost
    return {
        "availability": flown / total,
        "failures": failures,
        "planned": planned,
        "wasted_life": float(np.mean(wasted)) if wasted else 0.0,
        "cost": cost,
        "cost_per_1k": 1000 * cost / total,
        "failures_per_1k": 1000 * failures / total,
    }


def run_many(tracks, pool, policy, capacity, cfg, seeds) -> dict:
    runs = [simulate(tracks, draw_schedule(pool, tracks, cfg, s), policy, capacity, cfg) for s in seeds]
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for k in ("cost_per_1k", "failures_per_1k", "availability"):
        out[f"{k}_sd"] = float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0
    return out
