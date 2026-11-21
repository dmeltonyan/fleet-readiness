"""Load NASA C-MAPSS turbofan data and split engines into disjoint sets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def load_train(raw_dir: Path, name: str) -> pd.DataFrame:
    """Run-to-failure trajectories with the true remaining life at every cycle."""
    df = read_raw(Path(raw_dir) / f"train_{name}.txt")
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last - df["cycle"]
    return df


def load_test(raw_dir: Path, name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Truncated trajectories plus NASA's true remaining life after the last observed cycle."""
    df = read_raw(Path(raw_dir) / f"test_{name}.txt")
    rul = pd.read_csv(Path(raw_dir) / f"RUL_{name}.txt", header=None).iloc[:, 0]
    rul.index = np.sort(df["unit"].unique())
    rul.name = "rul"
    return df, rul


def split_units(units, fractions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Shuffle engine IDs once and cut them into disjoint groups."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    names = list(fractions)
    cuts = np.cumsum([fractions[n] for n in names])
    cuts = np.round(cuts / cuts[-1] * len(units)).astype(int)
    out, start = {}, 0
    for name, stop in zip(names, cuts):
        out[name] = np.sort(units[start:stop])
        start = stop
    return out
