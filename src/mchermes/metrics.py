"""Risk metrics computed from simulated outcomes, one array per option."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

PERCENTILES = (5, 10, 50, 90, 95)
CRITERIA = ("expected_value", "risk_averse", "min_regret")


@dataclass(frozen=True)
class OptionStats:
    """Summary of one option's outcome distribution.

    ``mean_se`` is the Monte Carlo standard error of the mean: rerunning with another
    seed moves the mean by about this much. ``var_95`` is the 5th percentile of the
    outcome: you do at least this well with 95% confidence. ``cvar_95`` is the mean
    of the worst 5% of outcomes. Both are in outcome units, so a negative value is a
    loss.
    """

    mean: float
    mean_se: float
    median: float
    std: float
    min: float
    max: float
    p5: float
    p10: float
    p50: float
    p90: float
    p95: float
    prob_loss: float
    var_95: float
    cvar_95: float
    prob_best: float
    expected_regret: float

    def to_dict(self) -> dict[str, float]:
        """Plain ``dict`` of every metric."""
        return asdict(self)


def cvar(values: FloatArray, level: float = 0.95) -> float:
    """Mean of the worst ``1 - level`` share of outcomes (expected shortfall)."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    tail = max(1, int(np.ceil(values.size * (1 - level) - 1e-9)))
    if tail >= values.size:
        return float(values.mean())
    return float(np.partition(values, tail - 1)[:tail].mean())


def best_share(matrix: FloatArray) -> FloatArray:
    """Probability each row (option) is the best, ties split equally per sample."""
    winners = matrix == matrix.max(axis=0)
    shares: FloatArray = (winners / winners.sum(axis=0)).mean(axis=1)
    return shares


def expected_regret(matrix: FloatArray) -> FloatArray:
    """Mean shortfall of each row (option) against the best option in the same sample."""
    regret: FloatArray = (matrix.max(axis=0) - matrix).mean(axis=1)
    return regret


def summarize(outcomes: Mapping[str, FloatArray]) -> dict[str, OptionStats]:
    """Compute :class:`OptionStats` for every option."""
    names = list(outcomes)
    matrix = np.vstack([outcomes[n] for n in names])
    shares, regrets = best_share(matrix), expected_regret(matrix)
    stats = {}
    for i, name in enumerate(names):
        x = matrix[i]
        p5, p10, p50, p90, p95 = (float(v) for v in np.percentile(x, PERCENTILES))
        std = float(x.std(ddof=1)) if x.size > 1 else 0.0
        stats[name] = OptionStats(
            mean=float(x.mean()),
            mean_se=std / float(np.sqrt(x.size)),
            median=p50,
            std=std,
            min=float(x.min()),
            max=float(x.max()),
            p5=p5,
            p10=p10,
            p50=p50,
            p90=p90,
            p95=p95,
            prob_loss=float(np.mean(x < 0)),
            var_95=p5,
            cvar_95=cvar(x, 0.95),
            prob_best=float(shares[i]),
            expected_regret=float(regrets[i]),
        )
    return stats


def rank_by(stats: Mapping[str, OptionStats], criterion: str) -> list[str]:
    """Option names from best to worst under ``criterion`` (see :data:`CRITERIA`)."""
    keys: dict[str, Callable[[OptionStats], float]] = {
        "expected_value": lambda s: -s.mean,
        "risk_averse": lambda s: -s.p10,
        "min_regret": lambda s: s.expected_regret,
    }
    if criterion not in keys:
        raise ValueError(f"unknown criterion {criterion!r}; choose one of {', '.join(CRITERIA)}")
    key = keys[criterion]
    return sorted(stats, key=lambda name: key(stats[name]))


__all__ = ["CRITERIA", "PERCENTILES", "OptionStats", "cvar", "rank_by", "summarize"]
