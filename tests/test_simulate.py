import numpy as np

from src.policies import Policy, PredictedRUL, RiskBased, RunToFailure, TimeBased
from src.simulate import EngineTrack, SimConfig, draw_schedule, simulate


def _tracks(n=12, seed=0):
    rng = np.random.default_rng(seed)
    tracks = {}
    for u in range(n):
        life = int(rng.integers(120, 260))
        true = np.arange(life - 1, -1, -1, dtype=float)
        noisy = np.clip(true + rng.normal(0, 8, life), 0, None)
        tracks[u] = EngineTrack(life, noisy, np.clip(noisy - 15, 0, None), noisy, noisy + 15)
    return tracks


CFG = SimConfig(fleet_size=8, horizon=1500, check_interval=10, scheduled_downtime=5,
                unscheduled_downtime=25, scheduled_cost=1.0, unscheduled_cost=6.0, initial_age_max=0.6)


class CountingPolicy(Policy):
    """Asks to pull every engine, and records how many were pulled per check via the simulator."""
    name = "everything"

    def decide(self, age, point, lower, median, upper):
        return True, 0.0


def test_capacity_is_never_exceeded():
    tracks = _tracks()
    sched = draw_schedule(list(tracks), tracks, CFG, seed=1)
    for cap in (1, 2, 3):
        out = simulate(tracks, sched, CountingPolicy(), cap, CFG)
        checks = CFG.horizon // CFG.check_interval
        assert out["planned"] <= cap * checks


def test_run_to_failure_never_plans_and_does_fail():
    tracks = _tracks()
    out = simulate(tracks, draw_schedule(list(tracks), tracks, CFG, seed=2), RunToFailure(), 30, CFG)
    assert out["planned"] == 0
    assert out["failures"] > 0


def test_availability_is_a_fraction_and_accounts_for_downtime():
    tracks = _tracks()
    sched = draw_schedule(list(tracks), tracks, CFG, seed=3)
    for pol in (RunToFailure(), TimeBased(param=150), PredictedRUL(param=20), RiskBased(param=20)):
        out = simulate(tracks, sched, pol, 30, CFG)
        assert 0 < out["availability"] < 1
        # Every failure and planned removal takes the position out of service for its downtime,
        # except possibly the last few at the very end of the horizon.
        lost = CFG.fleet_size * CFG.horizon * (1 - out["availability"])
        expected = out["failures"] * CFG.unscheduled_downtime + out["planned"] * CFG.scheduled_downtime
        assert lost <= expected + 1e-9
        assert lost >= expected - CFG.fleet_size * CFG.unscheduled_downtime


def test_same_schedule_same_result():
    tracks = _tracks()
    sched = draw_schedule(list(tracks), tracks, CFG, seed=4)
    a = simulate(tracks, sched, RiskBased(param=20), 2, CFG)
    b = simulate(tracks, sched, RiskBased(param=20), 2, CFG)
    assert a == b


def test_perfect_foresight_beats_running_to_failure():
    """With exact remaining life, a sensible threshold should avoid almost every failure."""
    rng = np.random.default_rng(5)
    tracks = {}
    for u in range(12):
        life = int(rng.integers(120, 260))
        true = np.arange(life - 1, -1, -1, dtype=float)
        tracks[u] = EngineTrack(life, true, true, true, true)
    sched = draw_schedule(list(tracks), tracks, CFG, seed=5)
    smart = simulate(tracks, sched, PredictedRUL(param=12), 30, CFG)
    naive = simulate(tracks, sched, RunToFailure(), 30, CFG)
    assert smart["failures"] < 0.1 * naive["failures"]
    assert smart["cost"] < naive["cost"]
