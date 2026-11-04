"""Simulation results: metrics, recommendations, tables, charts and exports."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mchermes.metrics import OptionStats, rank_by, summarize
from mchermes.sensitivity import Sensitivity, tornado

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure
    from rich.console import Console

FloatArray = NDArray[np.float64]


class Result:
    """Outcome samples for every option of a simulated :class:`~mchermes.Decision`."""

    def __init__(
        self,
        name: str,
        n: int,
        seed: int,
        criterion: str,
        inputs: dict[str, FloatArray],
        outcomes: dict[str, FloatArray],
    ) -> None:
        self.name, self.n, self.seed, self.criterion = name, n, seed, criterion
        self.inputs = inputs
        self.outcomes = outcomes

    @property
    def options(self) -> list[str]:
        """Option names in declaration order."""
        return list(self.outcomes)

    @cached_property
    def stats(self) -> dict[str, OptionStats]:
        """:class:`~mchermes.metrics.OptionStats` per option."""
        return summarize(self.outcomes)

    def _outcome(self, option: str) -> FloatArray:
        if option not in self.outcomes:
            raise KeyError(f"unknown option {option!r}; options are {', '.join(self.options)}")
        return self.outcomes[option]

    def best_option(self, criterion: str | None = None) -> str:
        """Best option under ``criterion``.

        ``"expected_value"`` maximizes the mean, ``"risk_averse"`` maximizes P10 and
        ``"min_regret"`` minimizes expected regret. Defaults to the decision's criterion.
        """
        return rank_by(self.stats, criterion or self.criterion)[0]

    def prob_loss(self, option: str, threshold: float = 0.0) -> float:
        """Probability that ``option``'s outcome is below ``threshold`` (default 0)."""
        return float(np.mean(self._outcome(option) < threshold))

    def percentile(self, option: str, q: float | Sequence[float]) -> Any:
        """Percentile(s) ``q`` in ``[0, 100]`` of ``option``'s outcome."""
        values = np.percentile(self._outcome(option), q)
        return float(values) if np.ndim(values) == 0 else values

    def sensitivity(self, option: str) -> list[Sensitivity]:
        """Inputs ranked by |Spearman correlation| with ``option``'s outcome."""
        return tornado(self.inputs, self._outcome(option))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary: metrics, recommendation and sensitivities."""
        return {
            "name": self.name,
            "samples": self.n,
            "seed": self.seed,
            "criterion": self.criterion,
            "best_option": self.best_option(),
            "options": {
                name: {
                    **stats.to_dict(),
                    "sensitivity": {s.variable: s.spearman for s in self.sensitivity(name)},
                }
                for name, stats in self.stats.items()
            },
        }

    def summary(self, console: Console | None = None) -> None:
        """Print a rich table of every metric and the recommendation."""
        from mchermes.report import print_summary

        print_summary(self, console)

    def plot(self, path: str | None = None, *, theme: str = "light", **kwargs: Any) -> Figure:
        """Overlay of outcome distributions; saved to ``path`` when given."""
        from mchermes.plotting import plot_distributions

        return plot_distributions(self, path, theme=theme, **kwargs)

    def plot_tornado(
        self,
        option: str | None = None,
        path: str | None = None,
        *,
        theme: str = "light",
        **kwargs: Any,
    ) -> Figure:
        """Tornado chart of input sensitivities for ``option`` (default: best option)."""
        from mchermes.plotting import plot_tornado

        name = option or self.best_option()
        self._outcome(name)
        return plot_tornado(self, name, path, theme=theme, **kwargs)

    def plot_cdf(self, path: str | None = None, *, theme: str = "light") -> Figure:
        """Cumulative distribution (S-curve) of every option."""
        from mchermes.plotting import plot_cdf

        return plot_cdf(self, path, theme=theme)

    def to_dataframe(self, kind: str = "samples") -> pd.DataFrame:
        """``"samples"``: one row per sample (inputs, then options); ``"summary"``:
        one row per option with every metric. Requires ``pandas``."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - exercised only without pandas
            raise ImportError(
                "to_dataframe() needs pandas: pip install 'montecarlo-hermes[pandas]'"
            ) from None
        if kind == "samples":
            frame: pd.DataFrame = pd.DataFrame({**self.inputs, **self.outcomes})
            return frame
        if kind == "summary":
            rows = [{"option": k, **v.to_dict()} for k, v in self.stats.items()]
            summary: pd.DataFrame = pd.DataFrame(rows).set_index("option")
            return summary
        raise ValueError(f"kind must be 'samples' or 'summary', got {kind!r}")

    def __repr__(self) -> str:
        return (
            f"Result({self.name!r}, n={self.n}, seed={self.seed}, "
            f"best_option={self.best_option()!r})"
        )


__all__ = ["Result"]
