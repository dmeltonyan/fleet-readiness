"""Maintenance policies. Each one answers, at every check: pull this engine? and how urgently?

Lower priority values are served first when there are more requests than maintenance slots.
"""
from __future__ import annotations

from dataclasses import dataclass


class Policy:
    name = "policy"
    param = None

    def decide(self, age, point, lower, median, upper) -> tuple[bool, float]:
        raise NotImplementedError


@dataclass
class RunToFailure(Policy):
    name: str = "Run to failure"

    def decide(self, age, point, lower, median, upper):
        return False, 0.0


@dataclass
class TimeBased(Policy):
    """Replace at a fixed age, the way most scheduled-maintenance programmes work. Oldest first."""

    param: float = 200
    name: str = "Time-based"

    def decide(self, age, point, lower, median, upper):
        return age >= self.param, -age


@dataclass
class PredictedRUL(Policy):
    """Pull when the model's point estimate of remaining life drops below a threshold."""

    param: float = 20
    name: str = "Predicted RUL"

    def decide(self, age, point, lower, median, upper):
        return point <= self.param, point


@dataclass
class RiskBased(Policy):
    """Pull when the pessimistic end of the calibrated band drops below a threshold.

    The lower bound is roughly the 10th percentile of remaining life, so the trigger reads
    "there is a real chance this engine fails before the next couple of checks". Ranking by
    the lower bound sends the engines most likely to fail soon through first when slots are short.
    """

    param: float = 20
    name: str = "Risk-based"

    def decide(self, age, point, lower, median, upper):
        return lower <= self.param, lower
