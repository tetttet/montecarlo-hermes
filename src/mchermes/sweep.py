"""What-if sweeps: how the decision changes as one input takes different values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from mchermes.result import Result

FloatArray = NDArray[np.float64]


class Sweep:
    """Results of re-running a decision with one variable fixed at several values."""

    def __init__(self, variable: str, values: list[float], results: list[Result]) -> None:
        self.variable, self.values, self.results = variable, values, results

    @property
    def options(self) -> list[str]:
        """Option names in declaration order."""
        return self.results[0].options

    def means(self, option: str) -> FloatArray:
        """Expected outcome of ``option`` at each swept value."""
        return np.array([r.stats[option].mean for r in self.results])

    def percentiles(self, option: str, q: float) -> FloatArray:
        """Percentile ``q`` of ``option``'s outcome at each swept value."""
        return np.array([r.percentile(option, q) for r in self.results])

    def best_options(self, criterion: str | None = None) -> list[str]:
        """Best option at each swept value."""
        return [r.best_option(criterion) for r in self.results]

    def breakeven(self, a: str, b: str) -> float | None:
        """Swept value where ``a`` and ``b`` have equal expected outcome.

        Linear interpolation between the two grid points around the first sign change
        of ``mean(a) - mean(b)``; ``None`` when the ranking never flips on the grid.
        """
        x = np.asarray(self.values)
        gap = self.means(a) - self.means(b)
        exact = np.flatnonzero(gap == 0)
        if exact.size:
            return float(x[exact[0]])
        flips = np.flatnonzero(np.sign(gap[:-1]) != np.sign(gap[1:]))
        if not flips.size:
            return None
        i = int(flips[0])
        return float(x[i] - gap[i] * (x[i + 1] - x[i]) / (gap[i + 1] - gap[i]))

    def to_rows(self) -> list[dict[str, Any]]:
        """One dict per swept value: means per option and the best option."""
        rows = []
        for value, result in zip(self.values, self.results, strict=True):
            row: dict[str, Any] = {self.variable: value}
            row.update({f"mean_{o}": result.stats[o].mean for o in self.options})
            row["best_option"] = result.best_option()
            rows.append(row)
        return rows

    def to_dataframe(self) -> pd.DataFrame:
        """:meth:`to_rows` as a DataFrame indexed by the swept value. Requires pandas."""
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame(self.to_rows()).set_index(self.variable)
        return frame

    def plot(self, path: str | None = None, *, theme: str = "light") -> Figure:
        """Expected outcome per option against the swept value, with P10-P90 bands."""
        from mchermes.plotting import plot_sweep

        return plot_sweep(self, path, theme=theme)

    def __repr__(self) -> str:
        return f"Sweep({self.variable!r}, {len(self.values)} values, options={self.options})"


__all__ = ["Sweep"]
