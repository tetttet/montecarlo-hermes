"""The :class:`Decision` builder: uncertain inputs, options, and the simulation run."""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import ArrayLike, NDArray

from mchermes.correlation import check_coefficient, reorder
from mchermes.distributions import Distribution, Fixed
from mchermes.formula import Formula, compile_formula
from mchermes.metrics import CRITERIA
from mchermes.result import Result

if TYPE_CHECKING:
    from mchermes.sweep import Sweep

FloatArray = NDArray[np.float64]
OptionSpec = Callable[[], ArrayLike] | str | float | int
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def stream(seed: int, name: str) -> np.random.Generator:
    """Random generator owned by one variable, derived from ``(seed, name)``.

    Each variable gets an independent stream, so adding, removing or reordering
    variables never changes the draws of the others.
    """
    sequence = np.random.SeedSequence(seed, spawn_key=(zlib.crc32(name.encode()),))
    return np.random.Generator(np.random.PCG64(sequence))


class Variable(NDArrayOperatorsMixin):
    """Handle to an uncertain input that behaves like a NumPy array inside formulas.

    Arithmetic, comparisons and NumPy ufuncs (``np.maximum``, ``np.exp``...) work on
    all samples at once. Values exist only while (and after) :meth:`Decision.run`
    evaluates the options.
    """

    def __init__(self, decision: Decision, name: str, distribution: Distribution) -> None:
        self._decision, self.name, self.distribution = decision, name, distribution

    @property
    def values(self) -> FloatArray:
        """Samples from the latest run of the owning decision."""
        samples = self._decision._samples
        if samples is None:
            raise RuntimeError(
                f"variable {self.name!r} has no samples yet: use it inside an option "
                "and call Decision.run()"
            )
        return samples[self.name]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> FloatArray:
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        unwrapped = [x.values if isinstance(x, Variable) else x for x in inputs]
        return getattr(ufunc, method)(*unwrapped, **kwargs)

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"Variable({self.name!r}, {self.distribution!r})"


class Decision:
    """A choice between options whose outcomes depend on uncertain inputs.

    Args:
        name: Human-readable question, e.g. ``"Open a second coffee shop?"``.
        n: Number of Monte Carlo samples.
        seed: Seed for reproducible results; ``None`` draws a fresh one (recorded
            on the result so the run can be repeated).
        criterion: Default rule for :meth:`Result.best_option`.
    """

    def __init__(
        self,
        name: str = "Decision",
        *,
        n: int = 100_000,
        seed: int | None = None,
        criterion: str = "expected_value",
    ) -> None:
        self.name = name
        self.n = _check_samples(n)
        self.seed = _check_seed(seed)
        self.criterion = _check_criterion(criterion)
        self.variables: dict[str, Variable] = {}
        self.options: dict[str, OptionSpec] = {}
        self._formulas: dict[str, Formula] = {}
        self.correlations: dict[tuple[str, str], float] = {}
        self._samples: dict[str, FloatArray] | None = None

    def uncertain(self, name: str, distribution: Distribution) -> Variable:
        """Declare an uncertain input and return a handle to use in option formulas."""
        self._check_name(name, "variable")
        if not isinstance(distribution, Distribution):
            raise TypeError(
                f"variable {name!r} needs a distribution such as mc.Normal(...), "
                f"got {type(distribution).__name__}"
            )
        variable = Variable(self, name, distribution)
        self.variables[name] = variable
        return variable

    @classmethod
    def from_yaml(cls, path: str | Path) -> Decision:
        """Load a decision file; same as :func:`mchermes.load`."""
        from mchermes.loader import load

        return load(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        """Build from a parsed config; same as :func:`mchermes.from_dict`."""
        from mchermes.loader import from_dict

        return from_dict(data)

    def correlate(self, a: str | Variable, b: str | Variable, rho: float) -> Decision:
        """Make two inputs move together with Spearman rank correlation ``rho``.

        Each input keeps its own distribution exactly; only the pairing of samples
        changes (Iman-Conover). Use a negative ``rho`` for inputs that move apart.
        """
        names = []
        for item in (a, b):
            name = item.name if isinstance(item, Variable) else item
            if name not in self.variables:
                raise ValueError(f"unknown variable {name!r}; declare it with uncertain() first")
            names.append(name)
        first, second = sorted(names)
        self.correlations[(first, second)] = check_coefficient(first, second, rho)
        return self

    def option(self, name: str, outcome: OptionSpec) -> Decision:
        """Add an option whose outcome is a zero-argument callable, a formula or a number.

        A callable is evaluated once on whole sample arrays (use the handles returned
        by :meth:`uncertain`). A string is a safe formula over variable names, e.g.
        ``"sales * margin - rent"``.
        """
        self._check_name(name, "option")
        if isinstance(outcome, str):
            self._formulas[name] = compile_formula(outcome, self.variables)
        elif isinstance(outcome, bool) or not (
            callable(outcome) or isinstance(outcome, (int, float))
        ):
            raise TypeError(
                f"option {name!r} needs a callable, a formula string or a number, "
                f"got {type(outcome).__name__}"
            )
        self.options[name] = outcome
        return self

    def run(self, n: int | None = None, seed: int | None = None) -> Result:
        """Sample every input, evaluate every option and return a :class:`Result`."""
        if not self.options:
            raise ValueError("add at least one option with Decision.option() before run()")
        n = self.n if n is None else _check_samples(n)
        seed = self.seed if seed is None else _check_seed(seed)
        if seed is None:
            seed = _fresh_seed()
        self._samples = self._sample_inputs(n, seed)
        outcomes = {name: self._evaluate(name, n) for name in self.options}
        return Result(self.name, n, seed, self.criterion, dict(self._samples), outcomes)

    def sweep(
        self, variable: str | Variable, values: Iterable[float], *, n: int | None = None
    ) -> Sweep:
        """Re-run the decision with ``variable`` fixed at each of ``values``.

        Every run reuses the same seed (common random numbers), so differences between
        runs come from the swept value, not from sampling noise.
        """
        from mchermes.sweep import Sweep

        name = variable.name if isinstance(variable, Variable) else variable
        if name not in self.variables:
            raise ValueError(f"unknown variable {name!r}")
        grid = [float(v) for v in values]
        if not grid:
            raise ValueError("sweep needs at least one value")
        handle = self.variables[name]
        original, correlations = handle.distribution, self.correlations
        seed = self.seed if self.seed is not None else _fresh_seed()
        # A fixed value cannot be correlated: drop its pairs for the sweep.
        self.correlations = {pair: r for pair, r in correlations.items() if name not in pair}
        results = []
        try:
            for value in grid:
                handle.distribution = Fixed(value)
                results.append(self.run(n=n, seed=seed))
        finally:
            handle.distribution, self.correlations = original, correlations
        return Sweep(name, grid, results)

    def _sample_inputs(self, n: int, seed: int) -> dict[str, FloatArray]:
        """Draw every variable from its own stream, then apply correlations."""
        samples = {}
        for name, variable in self.variables.items():
            values = variable.distribution.sample(stream(seed, name), n)
            values.setflags(write=False)
            samples[name] = values
        if self.correlations:
            samples.update(reorder(samples, self.correlations, stream(seed, "__correlations__")))
        return samples

    def _evaluate(self, name: str, n: int) -> FloatArray:
        spec = self.options[name]
        assert self._samples is not None
        if name in self._formulas:
            raw: Any = self._formulas[name].evaluate(self._samples)
        elif callable(spec):
            raw = spec()
        else:
            raw = spec
        if isinstance(raw, Variable):
            raw = raw.values
        try:
            values = np.asarray(raw, dtype=np.float64)
            values = np.array(np.broadcast_to(values, (n,)))
        except (TypeError, ValueError):
            raise ValueError(
                f"option {name!r} must produce a number or one value per sample "
                f"(shape ({n},)), got {np.shape(raw)}"
            ) from None
        if not np.all(np.isfinite(values)):
            bad = int(np.sum(~np.isfinite(values)))
            raise ValueError(
                f"option {name!r} produced {bad} non-finite values (NaN or inf); "
                "check for division by zero or log/sqrt of negative numbers"
            )
        return values

    def _check_name(self, name: str, kind: str) -> None:
        if not isinstance(name, str) or not NAME.match(name):
            raise ValueError(
                f"{kind} name {name!r} must be an identifier (letters, digits, underscores)"
            )
        if name in self.variables or name in self.options:
            raise ValueError(f"name {name!r} is already used in this decision")

    def __repr__(self) -> str:
        return (
            f"Decision({self.name!r}, n={self.n}, variables={list(self.variables)}, "
            f"options={list(self.options)})"
        )


def _fresh_seed() -> int:
    return int(np.random.SeedSequence().generate_state(1, np.uint64)[0])


def _check_samples(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n (number of samples) must be an integer >= 2, got {n!r}")
    return int(n)


def _check_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}")
    return int(seed)


def _check_criterion(criterion: str) -> str:
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; choose one of {', '.join(CRITERIA)}")
    return criterion


__all__ = ["Decision", "Variable", "stream"]
