import numpy as np
import pandas as pd
import pytest

from src.data import SENSORS, SETTINGS, split_units
from src.features import RegimeNormalizer, build_features


def _fake_fleet(n_units=4, n_cycles=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            regime = rng.integers(0, 3)
            settings = [regime * 10.0, regime * 0.1, 100.0]
            sensors = list(rng.normal(regime * 5 + c * 0.05, 1.0, len(SENSORS)))
            rows.append([u, c, *settings, *sensors])
    return pd.DataFrame(rows, columns=["unit", "cycle", *SETTINGS, *SENSORS])


def _features(df, norm):
    return build_features(norm.transform(df), norm.sensors_, [5, 10, 20], 20)


def test_features_never_look_ahead():
    """Appending later cycles to an engine must not change any feature at earlier cycles."""
    full = _fake_fleet()
    norm = RegimeNormalizer(3).fit(full)
    cut = full[full["cycle"] <= 35]
    a = _features(cut, norm)
    b = _features(full, norm)
    b = b[b["cycle"] <= 35].reset_index(drop=True)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b)


def test_features_do_not_leak_across_engines():
    """Changing one engine's data must leave every other engine's features untouched."""
    full = _fake_fleet()
    norm = RegimeNormalizer(3).fit(full)
    altered = full.copy()
    altered.loc[altered["unit"] == 2, SENSORS] += 50
    a = _features(full, norm)
    b = _features(altered, norm)
    keep = a["unit"] != 2
    pd.testing.assert_frame_equal(a[keep].reset_index(drop=True), b[keep].reset_index(drop=True))


def test_split_is_disjoint_and_complete():
    units = np.arange(1, 250)
    groups = split_units(units, {"train": 0.55, "calibrate": 0.15, "tune": 0.15, "test": 0.15}, seed=7)
    seen = np.concatenate(list(groups.values()))
    assert len(seen) == len(set(seen)) == len(units)
    for a in groups:
        for b in groups:
            if a < b:
                assert not set(groups[a]) & set(groups[b])


def test_regime_normalisation_removes_operating_condition():
    df = _fake_fleet(n_units=6, n_cycles=80)
    norm = RegimeNormalizer(3).fit(df)
    z = norm.transform(df)
    per_regime = z.groupby("regime")[norm.sensors_].mean().abs().to_numpy()
    assert per_regime.max() < 0.3
