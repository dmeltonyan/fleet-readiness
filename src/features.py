"""Leak-free features: every value at cycle t uses only cycles <= t of the same engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import SENSORS, SETTINGS


@dataclass
class RegimeNormalizer:
    """Flight conditions shift sensor readings far more than wear does.

    Cluster the three operating settings into regimes, then z-score each sensor
    within its regime so what remains is degradation, not altitude or throttle.
    Fitted on training engines only.
    """

    n_regimes: int
    seed: int = 7

    def fit(self, df: pd.DataFrame) -> "RegimeNormalizer":
        settings = df[SETTINGS].round(2).to_numpy()
        k = min(self.n_regimes, len(np.unique(settings, axis=0)))
        self.kmeans_ = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(settings)
        regimes = self.kmeans_.predict(settings)
        stats = df[SENSORS].assign(regime=regimes).groupby("regime")
        self.mean_ = stats.mean()
        self.std_ = stats.std().replace(0, np.nan)
        # Sensors that never move within a regime carry no information.
        spread = (self.std_.fillna(0) > 1e-6).mean()
        self.sensors_ = [s for s in SENSORS if spread[s] >= 0.5]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        regimes = self.kmeans_.predict(df[SETTINGS].round(2).to_numpy())
        out = df[["unit", "cycle"]].copy()
        out["regime"] = regimes
        mean = self.mean_.loc[regimes, self.sensors_].to_numpy()
        std = self.std_.loc[regimes, self.sensors_].to_numpy()
        z = (df[self.sensors_].to_numpy() - mean) / std
        out[self.sensors_] = np.nan_to_num(z)
        return out


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over the trailing window (partial windows allowed), from rolling sums."""
    i = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    r = lambda v: v.rolling(window, min_periods=1).sum()
    n = x.rolling(window, min_periods=1).count()
    si, sx, sxi, sii = r(i), r(x), r(x * i), r(i * i)
    denom = n * sii - si**2
    slope = (n * sxi - si * sx) / denom.replace(0, np.nan)
    return slope.where(n >= 3, 0.0).fillna(0.0)


def build_features(norm: pd.DataFrame, sensors: list[str], windows: list[int], slope_window: int) -> pd.DataFrame:
    """Trailing-window summaries per engine. Rolling and ewm only look backwards."""
    norm = norm.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = norm.groupby("unit", sort=False)
    feats = {"cycle": norm["cycle"].to_numpy()}
    for s in sensors:
        col = g[s]
        for w in windows:
            feats[f"{s}_mean{w}"] = col.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).to_numpy()
        feats[f"{s}_ewm"] = col.transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean()).to_numpy()
        feats[f"{s}_slope"] = col.transform(lambda x: _rolling_slope(x, slope_window)).to_numpy()
        feats[f"{s}_drift"] = (
            col.transform(lambda x: x.ewm(alpha=0.1, adjust=False).mean())
            - col.transform(lambda x: x.expanding(min_periods=1).mean().iloc[:20].reindex(x.index).ffill())
        ).to_numpy()
    out = pd.DataFrame(feats)
    out.insert(0, "unit", norm["unit"].to_numpy())
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("unit", "rul")]
