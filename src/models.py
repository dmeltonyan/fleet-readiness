"""Remaining-useful-life models: a point estimate plus a calibrated uncertainty band."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """PHM08 challenge score. Late predictions (pred > true) are punished harder than early ones,
    because telling a maintainer an engine has more life than it does is the dangerous mistake."""
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


@dataclass
class FleetLifeBaseline:
    """What a time-based programme knows: the average engine life, minus this engine's age."""

    cap: float
    mean_life_: float = 0.0

    def fit(self, cycles: np.ndarray, rul: np.ndarray) -> "FleetLifeBaseline":
        self.mean_life_ = float(np.mean(cycles + rul))
        return self

    def predict(self, cycles: np.ndarray) -> np.ndarray:
        return np.clip(self.mean_life_ - cycles, 0, self.cap)


def ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


@dataclass
class RULModel:
    """LightGBM median plus lower/upper quantiles, widened by split-conformal calibration so the
    band actually covers the stated share of held-out engines."""

    params: dict
    quantiles: list[float]
    coverage: float
    models_: dict = field(default_factory=dict)
    margin_: float = 0.0

    def fit(self, X, y) -> "RULModel":
        self.point_ = lgb.LGBMRegressor(objective="regression", **self.params).fit(X, y)
        for q in self.quantiles:
            self.models_[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params).fit(X, y)
        return self

    def calibrate(self, X, y) -> "RULModel":
        lo, hi = self._raw_band(X)
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.margin_ = float(np.quantile(scores, level))
        return self

    def _raw_band(self, X):
        qs = sorted(self.quantiles)
        return self.models_[qs[0]].predict(X), self.models_[qs[-1]].predict(X)

    def predict(self, X) -> np.ndarray:
        return np.clip(self.point_.predict(X), 0, None)

    def predict_band(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self._raw_band(X)
        med = self.models_[0.5].predict(X) if 0.5 in self.models_ else self.predict(X)
        return (
            np.clip(lo - self.margin_, 0, None),
            np.clip(med, 0, None),
            np.clip(hi + self.margin_, 0, None),
        )
