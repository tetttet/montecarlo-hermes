"""Correlated inputs by rank reordering (Iman-Conover).

Each variable keeps exactly the values it would draw on its own; only their pairing
across variables changes, so marginal distributions are untouched. Targets are
Spearman rank correlations, which work for any mix of distributions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class CorrelationError(ValueError):
    """Correlation targets that are out of range or mutually impossible."""


def check_coefficient(a: str, b: str, rho: float) -> float:
    """Validated target rank correlation between two distinct variables."""
    if a == b:
        raise CorrelationError(f"cannot correlate {a!r} with itself")
    try:
        value = float(rho)
    except (TypeError, ValueError):
        raise CorrelationError(f"correlation {a!r}-{b!r} must be a number") from None
    if not -1 < value < 1:
        raise CorrelationError(
            f"correlation {a!r}-{b!r} must be strictly between -1 and 1, got {rho!r}"
        )
    return value


def target_matrix(names: list[str], pairs: Mapping[tuple[str, str], float]) -> FloatArray:
    """Pearson correlation of normal scores giving the requested Spearman targets."""
    index = {name: i for i, name in enumerate(names)}
    matrix = np.eye(len(names))
    for (a, b), rho in pairs.items():
        # Spearman rho_s of a bivariate normal with Pearson r is (6/pi) asin(r/2).
        r = 2 * math.sin(math.pi * rho / 6)
        matrix[index[a], index[b]] = matrix[index[b], index[a]] = r
    if np.min(np.linalg.eigvalsh(matrix)) <= 1e-10:
        raise CorrelationError(
            "the correlations are mutually inconsistent (the matrix is not positive "
            "definite); weaken some of them"
        )
    return matrix


def _normalized_scores(rng: np.random.Generator, n: int, k: int) -> FloatArray:
    """Standard normal scores whose sample correlation is exactly the identity."""
    scores = rng.standard_normal((n, k))
    scores -= scores.mean(axis=0)
    sample = np.corrcoef(scores, rowvar=False).reshape(k, k)
    whitened: FloatArray = scores @ np.linalg.inv(np.linalg.cholesky(sample)).T
    return whitened


def reorder(
    samples: Mapping[str, FloatArray],
    pairs: Mapping[tuple[str, str], float],
    rng: np.random.Generator,
) -> dict[str, FloatArray]:
    """Rearrange the samples of correlated variables to match ``pairs`` targets."""
    names = sorted({name for pair in pairs for name in pair})
    for name in names:
        if np.ptp(samples[name]) == 0:
            raise CorrelationError(f"cannot correlate {name!r}: it never varies")
    n = len(samples[names[0]])
    matrix = target_matrix(names, pairs)
    scores = _normalized_scores(rng, n, len(names)) @ np.linalg.cholesky(matrix).T
    result = {}
    for j, name in enumerate(names):
        values = np.empty(n)
        values[np.argsort(scores[:, j], kind="stable")] = np.sort(samples[name])
        values.setflags(write=False)
        result[name] = values
    return result


__all__ = ["CorrelationError", "check_coefficient", "reorder", "target_matrix"]
