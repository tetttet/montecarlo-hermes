"""Sensitivity of an outcome to each uncertain input, from the samples already drawn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Sensitivity:
    """Spearman rank correlation of one input with an option's outcome."""

    variable: str
    spearman: float

    @property
    def impact(self) -> float:
        """Strength of the association regardless of direction."""
        return abs(self.spearman)


def rank(values: FloatArray) -> FloatArray:
    """Ranks starting at 1, ties receiving the average of their ranks."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    averages: FloatArray = ends - (counts - 1) / 2
    return averages[inverse.reshape(-1)]


def spearman(x: FloatArray, y: FloatArray) -> float:
    """Spearman rank correlation; 0.0 when either side is constant."""
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = float(np.sqrt((rx @ rx) * (ry @ ry)))
    return float(rx @ ry / denominator) if denominator > 0 else 0.0


def tornado(inputs: Mapping[str, FloatArray], outcome: FloatArray) -> list[Sensitivity]:
    """Rank correlation of every non-constant input with ``outcome``, by impact."""
    rows = [
        Sensitivity(name, spearman(values, outcome))
        for name, values in inputs.items()
        if np.ptp(values) > 0
    ]
    return sorted(rows, key=lambda r: -r.impact)


__all__ = ["Sensitivity", "rank", "spearman", "tornado"]
