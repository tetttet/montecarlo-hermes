"""Sensitivity of an outcome to each uncertain input, from the samples already drawn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Rank correlations are estimated on at most this many samples: the standard error is
# then below 0.003, and ranking stays fast for multi-million-sample runs.
MAX_SAMPLES = 200_000


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
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    if starts.size == values.size:  # no ties: the common case for continuous inputs
        ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
        return ranks
    ends = np.r_[starts[1:], values.size]
    averages = (starts + ends + 1) / 2
    ranks[order] = np.repeat(averages, ends - starts)
    return ranks


def centered_ranks(values: FloatArray) -> FloatArray:
    """Ranks of the first :data:`MAX_SAMPLES` values, centered on zero."""
    r = rank(np.asarray(values[:MAX_SAMPLES], dtype=np.float64))
    centered: FloatArray = r - r.mean()
    return centered


def _correlation(rx: FloatArray, ry: FloatArray) -> float:
    denominator = float(np.sqrt((rx @ rx) * (ry @ ry)))
    return float(rx @ ry / denominator) if denominator > 0 else 0.0


def spearman(x: FloatArray, y: FloatArray) -> float:
    """Spearman rank correlation; 0.0 when either side is constant."""
    return _correlation(centered_ranks(x), centered_ranks(y))


def tornado(
    inputs: Mapping[str, FloatArray],
    outcome: FloatArray,
    ranks: Mapping[str, FloatArray] | None = None,
) -> list[Sensitivity]:
    """Rank correlation of every non-constant input with ``outcome``, by impact.

    ``ranks`` may hold precomputed :func:`centered_ranks` of the inputs (reused across
    options). Empty when the outcome itself is constant: nothing moves it.
    """
    if np.ptp(outcome) == 0:
        return []
    ry = centered_ranks(outcome)
    rows = []
    for name, values in inputs.items():
        if np.ptp(values) == 0:
            continue
        rx = ranks[name] if ranks is not None and name in ranks else centered_ranks(values)
        rows.append(Sensitivity(name, _correlation(rx, ry)))
    return sorted(rows, key=lambda r: -r.impact)


__all__ = ["MAX_SAMPLES", "Sensitivity", "centered_ranks", "rank", "spearman", "tornado"]
