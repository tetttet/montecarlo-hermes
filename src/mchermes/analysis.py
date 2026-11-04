"""Decision analysis beyond summary metrics: value of information and comparisons.

Value of information answers "how much would it be worth to know X before deciding?".
EVPI prices knowing everything; EVPPI prices learning one input. Both are in outcome
units and assume the expected-value criterion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from mchermes.sensitivity import MAX_SAMPLES

FloatArray = NDArray[np.float64]

MAX_BINS = 50
MIN_PER_BIN = 200


@dataclass(frozen=True)
class InformationValue:
    """Expected gain from learning ``variable`` before choosing (EVPPI)."""

    variable: str
    value: float
    share_of_evpi: float


@dataclass(frozen=True)
class Comparison:
    """Distribution of ``a - b`` across simulated futures."""

    a: str
    b: str
    prob_a_better: float
    mean_difference: float
    p5: float
    p50: float
    p95: float

    def to_dict(self) -> dict[str, float | str]:
        """Plain ``dict`` of every field."""
        return asdict(self)


def evpi(matrix: FloatArray) -> float:
    """Expected value of perfect information: E[max over options] - max E[option]."""
    return max(float(np.mean(np.max(matrix, axis=0)) - np.max(np.mean(matrix, axis=1))), 0.0)


def _groups(x: FloatArray) -> NDArray[np.intp]:
    """Group label per sample: one group per value when there are few distinct values,
    else equal-count quantile bins (tied values always share a bin)."""
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    new_value = np.r_[True, ordered[1:] != ordered[:-1]]
    group = np.cumsum(new_value) - 1
    bins = max(2, min(MAX_BINS, x.size // MIN_PER_BIN))
    if group[-1] + 1 > bins:
        first_position = np.flatnonzero(new_value)[group]
        group = first_position * bins // x.size
    labels = np.empty(x.size, dtype=np.intp)
    labels[order] = group
    return labels


def evppi(x: FloatArray, matrix: FloatArray) -> float:
    """Expected value of learning ``x`` exactly before choosing (binned estimate).

    Uses the first :data:`~mchermes.sensitivity.MAX_SAMPLES` samples.
    """
    x, matrix = x[:MAX_SAMPLES], matrix[:, :MAX_SAMPLES]
    labels = _groups(x)
    counts = np.bincount(labels).astype(np.float64)
    used = counts > 0
    sums = np.vstack([np.bincount(labels, weights=row) for row in matrix])[:, used]
    conditional_best = (sums / counts[used]).max(axis=0)
    baseline = float(np.max(np.mean(matrix, axis=1)))
    gain = float(conditional_best @ (counts[used] / x.size)) - baseline
    if gain <= 1e-9 * max(abs(baseline), 1.0):  # rounding noise, not information
        return 0.0
    return min(gain, evpi(matrix))


def information_values(
    inputs: Mapping[str, FloatArray], matrix: FloatArray
) -> list[InformationValue]:
    """EVPPI of every non-constant input, most valuable first."""
    total = evpi(matrix)
    rows = []
    for name, x in inputs.items():
        if np.ptp(x) == 0:
            continue
        value = evppi(x, matrix)
        rows.append(InformationValue(name, value, value / total if total > 0 else 0.0))
    return sorted(rows, key=lambda r: -r.value)


def compare(a: str, b: str, xa: FloatArray, xb: FloatArray) -> Comparison:
    """How ``a`` fares against ``b`` sample by sample."""
    diff = xa - xb
    p5, p50, p95 = (float(v) for v in np.percentile(diff, [5, 50, 95]))
    return Comparison(a, b, float(np.mean(diff > 0)), float(diff.mean()), p5, p50, p95)


__all__ = [
    "Comparison",
    "InformationValue",
    "compare",
    "evpi",
    "evppi",
    "information_values",
]
