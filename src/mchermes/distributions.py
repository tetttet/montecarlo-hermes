"""Probability distributions for uncertain inputs.

Every distribution validates its parameters on construction and draws a vectorized
sample with :meth:`Distribution.sample`. Analytic ``mean`` and ``std`` are exposed so
users (and tests) can sanity-check a model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


class Distribution(ABC):
    """Base class: a validated, immutable description of an uncertain quantity."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        """Draw ``n`` independent samples as a float64 array."""

    @property
    @abstractmethod
    def mean(self) -> float:
        """Analytic mean."""

    @property
    @abstractmethod
    def std(self) -> float:
        """Analytic standard deviation."""

    def _params(self) -> dict[str, object]:
        return {k.lstrip("_"): v for k, v in vars(self).items()}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._params().items())
        return f"{type(self).__name__}({args})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and repr(self) == repr(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Normal(Distribution):
    """Gaussian with the given ``mean`` and standard deviation ``std`` (> 0)."""

    def __init__(self, mean: float, std: float) -> None:
        self.mu = _finite("Normal mean", mean)
        self.sigma = _positive("Normal std", std)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.normal(self.mu, self.sigma, n)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        return self.sigma

    def _params(self) -> dict[str, object]:
        return {"mean": self.mu, "std": self.sigma}


class LogNormal(Distribution):
    """Positive, right-skewed quantity given by its own ``mean`` and ``std`` (both > 0).

    Use :meth:`from_log_params` to specify ``mu``/``sigma`` of the underlying normal.
    """

    def __init__(self, mean: float, std: float) -> None:
        m = _positive("LogNormal mean", mean)
        s = _positive("LogNormal std", std)
        self.sigma = math.sqrt(math.log1p((s / m) ** 2))
        self.mu = math.log(m) - self.sigma**2 / 2

    @classmethod
    def from_log_params(cls, mu: float, sigma: float) -> LogNormal:
        """Build from the mean ``mu`` and std ``sigma`` of ``log(X)``."""
        mu = _finite("LogNormal mu", mu)
        sigma = _positive("LogNormal sigma", sigma)
        mean = math.exp(mu + sigma**2 / 2)
        return cls(mean, mean * math.sqrt(math.expm1(sigma**2)))

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.lognormal(self.mu, self.sigma, n)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    @property
    def std(self) -> float:
        return self.mean * math.sqrt(math.expm1(self.sigma**2))

    def _params(self) -> dict[str, object]:
        return {"mean": round(self.mean, 12), "std": round(self.std, 12)}


class Uniform(Distribution):
    """Every value in ``[low, high)`` equally likely."""

    def __init__(self, low: float, high: float) -> None:
        self.low = _finite("Uniform low", low)
        self.high = _finite("Uniform high", high)
        if self.low >= self.high:
            raise ValueError(f"Uniform needs low < high, got low={low!r}, high={high!r}")

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.uniform(self.low, self.high, n)

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2

    @property
    def std(self) -> float:
        return (self.high - self.low) / math.sqrt(12)


def _check_three_point(kind: str, low: float, mode: float, high: float) -> tuple[float, ...]:
    lo = _finite(f"{kind} low", low)
    mo = _finite(f"{kind} mode", mode)
    hi = _finite(f"{kind} high", high)
    if not lo <= mo <= hi or lo == hi:
        raise ValueError(
            f"{kind} needs low <= mode <= high and low < high, "
            f"got low={low!r}, mode={mode!r}, high={high!r}"
        )
    return lo, mo, hi


class Triangular(Distribution):
    """Three-point estimate: minimum ``low``, most likely ``mode``, maximum ``high``."""

    def __init__(self, low: float, mode: float, high: float) -> None:
        self.low, self.mode, self.high = _check_three_point("Triangular", low, mode, high)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.triangular(self.low, self.mode, self.high, n)

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    @property
    def std(self) -> float:
        a, c, b = self.low, self.mode, self.high
        return math.sqrt((a * a + b * b + c * c - a * b - a * c - b * c) / 18)


class Beta(Distribution):
    """Beta(``alpha``, ``beta``) rescaled to ``[low, high]`` (default ``[0, 1]``)."""

    def __init__(self, alpha: float, beta: float, low: float = 0.0, high: float = 1.0) -> None:
        self.alpha = _positive("Beta alpha", alpha)
        self.beta = _positive("Beta beta", beta)
        self.low = _finite("Beta low", low)
        self.high = _finite("Beta high", high)
        if self.low >= self.high:
            raise ValueError(f"Beta needs low < high, got low={low!r}, high={high!r}")

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return self.low + (self.high - self.low) * rng.beta(self.alpha, self.beta, n)

    @property
    def mean(self) -> float:
        return self.low + (self.high - self.low) * self.alpha / (self.alpha + self.beta)

    @property
    def std(self) -> float:
        a, b = self.alpha, self.beta
        return (self.high - self.low) * math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


class PERT(Beta):
    """Smooth three-point estimate (Beta-PERT); ``lamb`` controls weight on the mode."""

    def __init__(self, low: float, mode: float, high: float, lamb: float = 4.0) -> None:
        lo, mo, hi = _check_three_point("PERT", low, mode, high)
        self.mode = mo
        self.lamb = _positive("PERT lamb", lamb)
        super().__init__(
            1 + self.lamb * (mo - lo) / (hi - lo), 1 + self.lamb * (hi - mo) / (hi - lo), lo, hi
        )

    def _params(self) -> dict[str, object]:
        return {"low": self.low, "mode": self.mode, "high": self.high, "lamb": self.lamb}


class Bernoulli(Distribution):
    """1.0 with probability ``p`` (the event happens), else 0.0."""

    def __init__(self, p: float) -> None:
        self.p = _finite("Bernoulli p", p)
        if not 0 <= self.p <= 1:
            raise ValueError(f"Bernoulli p must be in [0, 1], got {p!r}")

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return (rng.random(n) < self.p).astype(np.float64)

    @property
    def mean(self) -> float:
        return self.p

    @property
    def std(self) -> float:
        return math.sqrt(self.p * (1 - self.p))


class Discrete(Distribution):
    """One of ``values``, drawn with the matching ``probabilities`` (must sum to 1)."""

    def __init__(self, values: Sequence[float], probabilities: Sequence[float]) -> None:
        if isinstance(values, (str, bytes)) or isinstance(probabilities, (str, bytes)):
            raise ValueError("Discrete values and probabilities must be lists of numbers")
        self.values = tuple(_finite("Discrete value", v) for v in values)
        self.probabilities = tuple(_finite("Discrete probability", p) for p in probabilities)
        if not self.values:
            raise ValueError("Discrete needs at least one value")
        if len(self.values) != len(self.probabilities):
            raise ValueError(
                f"Discrete needs one probability per value, got {len(self.values)} values "
                f"and {len(self.probabilities)} probabilities"
            )
        if any(p < 0 for p in self.probabilities):
            raise ValueError("Discrete probabilities must be >= 0")
        total = math.fsum(self.probabilities)
        if abs(total - 1) > 1e-9:
            raise ValueError(f"Discrete probabilities must sum to 1, got {total:g}")

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        p = np.asarray(self.probabilities)
        return rng.choice(np.asarray(self.values), size=n, p=p / p.sum())

    @property
    def mean(self) -> float:
        return math.fsum(v * p for v, p in zip(self.values, self.probabilities, strict=True))

    @property
    def std(self) -> float:
        m = self.mean
        pairs = zip(self.values, self.probabilities, strict=True)
        return math.sqrt(math.fsum(p * (v - m) ** 2 for v, p in pairs))


class Fixed(Distribution):
    """A known constant, so certain and uncertain inputs mix freely."""

    def __init__(self, value: float) -> None:
        self.value = _finite("Fixed value", value)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return np.full(n, self.value)

    @property
    def mean(self) -> float:
        return self.value

    @property
    def std(self) -> float:
        return 0.0


__all__ = [
    "PERT",
    "Bernoulli",
    "Beta",
    "Discrete",
    "Distribution",
    "Fixed",
    "LogNormal",
    "Normal",
    "Triangular",
    "Uniform",
]
