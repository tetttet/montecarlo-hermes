"""Matplotlib charts: distribution overlay, tornado and cumulative (S-curve)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.typing import RcKeyType

    from mchermes.result import Result
    from mchermes.sweep import Sweep

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Theme:
    """Colors for one chart theme; series colors are assigned in fixed order."""

    surface: str
    text: str
    muted: str
    grid: str
    series: tuple[str, ...]


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        text="#0b0b0b",
        muted="#52514e",
        grid="#e6e5e1",
        series=(
            "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#6250d6", "#e34948",
        ),
    ),
    "dark": Theme(
        surface="#1a1a19",
        text="#f0efec",
        muted="#c3c2b7",
        grid="#33332f",
        series=(
            "#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767",
        ),
    ),
}  # fmt: skip


def get_theme(name: str) -> Theme:
    """Theme by name (``"light"`` or ``"dark"``)."""
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; choose one of {', '.join(THEMES)}")
    return THEMES[name]


def compact(value: float, _position: object = None) -> str:
    """Axis label like ``-12k`` or ``1.5M``."""
    magnitude = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            return f"{value / limit:.3g}{suffix}".replace("-", "−")
    return f"{value:.3g}".replace("-", "−")


@contextmanager
def _style(theme: Theme) -> Iterator[None]:
    params: dict[RcKeyType, Any] = {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.grid,
        "axes.labelcolor": theme.muted,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": theme.text,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
    with plt.rc_context(params):
        yield


def _header(ax: Axes, theme: Theme, title: str, subtitle: str, show: bool = True) -> None:
    if not show:
        return
    ax.set_title(title, loc="left", pad=28, color=theme.text)
    ax.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        color=theme.muted,
        fontsize=10,
    )


def _finish(fig: Figure, path: str | None) -> Figure:
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return fig


def density(values: FloatArray, grid: FloatArray) -> FloatArray:
    """Gaussian kernel density on an evenly spaced ``grid`` (binned, Silverman bandwidth)."""
    step = float(grid[1] - grid[0])
    edges = np.append(grid - step / 2, grid[-1] + step / 2)
    counts, _ = np.histogram(values, bins=edges)
    q75, q25 = np.percentile(values, [75, 25])
    spread = min(float(values.std()), float(q75 - q25) / 1.34) or float(values.std())
    bandwidth = 0.9 * spread * values.size ** (-0.2)
    sigma = max(bandwidth / step, 0.5)
    half = min(int(4 * sigma) + 1, grid.size - 1)
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed: FloatArray = np.convolve(counts, kernel, mode="same") / (values.size * step)
    return smoothed


def _visible_range(result: Result, names: list[str]) -> tuple[float, float]:
    """Outcome range covering P0.5-P99.5 of every option, padded by 6%."""
    lo = min(float(np.percentile(result.outcomes[n], 0.5)) for n in names)
    hi = max(float(np.percentile(result.outcomes[n], 99.5)) for n in names)
    pad = (hi - lo) * 0.06 or max(abs(lo), 1.0) * 0.5
    return lo - pad, hi + pad


def plot_distributions(
    result: Result,
    path: str | None = None,
    *,
    theme: str = "light",
    options: list[str] | None = None,
    title: str | None = None,
    header: bool = True,
) -> Figure:
    """Overlaid outcome densities with shaded P5-P95 bands and dashed means."""
    colors = get_theme(theme)
    names = options or result.options
    lo, hi = _visible_range(result, names)
    grid = np.linspace(lo, hi, 600)
    with _style(colors):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        if grid[0] < 0 < grid[-1]:
            ax.axvline(0, color=colors.muted, linewidth=1, alpha=0.6, zorder=1)
        for i, name in enumerate(names):
            color = colors.series[i % len(colors.series)]
            values, stats = result.outcomes[name], result.stats[name]
            label = f"{name}  ·  mean {compact(stats.mean)}"
            if stats.std == 0:
                ax.axvline(stats.mean, color=color, linewidth=2.5, label=f"{label} (certain)")
                continue
            y = density(values, grid)
            band = (grid >= stats.p5) & (grid <= stats.p95)
            ax.fill_between(grid, y, color=color, alpha=0.10, linewidth=0)
            ax.fill_between(grid, y, where=band, color=color, alpha=0.28, linewidth=0)
            ax.plot(grid, y, color=color, linewidth=2, label=label)
            ax.axvline(stats.mean, color=color, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Likelihood")
        _header(
            ax,
            colors,
            title or result.name,
            f"{result.n:,} simulations  ·  shaded band = P5–P95  ·  dashed line = mean",
            header,
        )
        ax.legend(loc="upper right", labelcolor=colors.text)
    return _finish(fig, path)


def plot_tornado(
    result: Result,
    option: str,
    path: str | None = None,
    *,
    theme: str = "light",
    top: int = 12,
    min_impact: float = 0.02,
    header: bool = True,
) -> Figure:
    """Horizontal bars of each input's rank correlation with ``option``'s outcome.

    Shows at most ``top`` inputs and hides those with ``|rho| < min_impact``.
    """
    colors = get_theme(theme)
    rows = [r for r in result.sensitivity(option) if r.impact >= min_impact][:top][::-1]
    with _style(colors):
        fig, ax = plt.subplots(figsize=(10, max(2.8, 1.1 + 0.55 * len(rows))))
        labels = [r.variable for r in rows]
        values = [r.spearman for r in rows]
        bar_colors = [colors.series[0] if v >= 0 else colors.series[1] for v in values]
        ax.barh(labels, values, color=bar_colors, height=0.6, zorder=2)
        for y, v in enumerate(values):
            ax.text(
                v + (0.02 if v >= 0 else -0.02),
                y,
                f"{v:+.2f}".replace("-", "−"),
                va="center",
                ha="left" if v >= 0 else "right",
                color=colors.text,
                fontsize=10,
            )
        if not rows:
            ax.text(0, 0, "No uncertain input moves this outcome", ha="center", va="center",
                    color=colors.muted)  # fmt: skip
            ax.set_yticks([])
        ax.axvline(0, color=colors.muted, linewidth=1, zorder=3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=colors.text)
        ax.set_xlabel("Spearman rank correlation with the outcome")
        _header(
            ax,
            colors,
            f"What drives “{option}”?",
            "Longer bar = bigger influence  ·  blue raises the outcome, orange lowers it",
            header,
        )
    return _finish(fig, path)


def plot_cdf(
    result: Result, path: str | None = None, *, theme: str = "light", header: bool = True
) -> Figure:
    """Cumulative probability (S-curve) of every option, with P(loss) markers."""
    colors = get_theme(theme)
    levels = np.linspace(0, 100, 801)
    with _style(colors):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        crosses_zero = False
        for i, name in enumerate(result.options):
            color = colors.series[i % len(colors.series)]
            values, stats = result.outcomes[name], result.stats[name]
            quantiles = np.percentile(values, levels)
            ax.plot(quantiles, levels / 100, color=color, linewidth=2.2, label=name)
            if stats.min < 0 < stats.max:
                crosses_zero = True
                ax.plot(
                    0,
                    stats.prob_loss,
                    "o",
                    markersize=8,
                    color=color,
                    markeredgecolor=colors.surface,
                    markeredgewidth=2,
                    zorder=4,
                )
                ax.annotate(
                    f"P(loss) {stats.prob_loss:.0%}",
                    (0, stats.prob_loss),
                    xytext=(8, -4),
                    textcoords="offset points",
                    color=colors.text,
                    fontsize=10,
                    va="top",
                )
        if crosses_zero:
            ax.axvline(0, color=colors.muted, linewidth=1, alpha=0.6)
        lo, hi = _visible_range(result, result.options)
        if hi > lo:
            ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Chance the outcome is at most x")
        _header(
            ax,
            colors,
            "Cumulative outcome (S-curve)",
            "Further right = better  ·  read P(outcome ≤ x) off the curve",
            header,
        )
        ax.legend(loc="lower right", labelcolor=colors.text)
    return _finish(fig, path)


def plot_sweep(sweep: Sweep, path: str | None = None, *, theme: str = "light") -> Figure:
    """Expected outcome of each option across the swept values, with P10-P90 bands."""
    colors = get_theme(theme)
    x = np.asarray(sweep.values)
    with _style(colors):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        for i, name in enumerate(sweep.options):
            color = colors.series[i % len(colors.series)]
            low, high = sweep.percentiles(name, 10), sweep.percentiles(name, 90)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, sweep.means(name), color=color, linewidth=2.2, marker="o", label=name)
        for a_index, a in enumerate(sweep.options):
            for b in sweep.options[a_index + 1 :]:
                point = sweep.breakeven(a, b)
                if point is not None:
                    ax.axvline(point, color=colors.muted, linewidth=1, linestyle=(0, (4, 3)))
                    ax.annotate(
                        f"breakeven {compact(point)}",
                        xy=(point, 1),
                        xycoords=("data", "axes fraction"),
                        xytext=(6, -14),
                        textcoords="offset points",
                        color=colors.text,
                        fontsize=10,
                    )
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.yaxis.set_major_formatter(FuncFormatter(compact))
        ax.set_xlabel(sweep.variable)
        ax.set_ylabel("Expected outcome")
        _header(
            ax,
            colors,
            f"What if {sweep.variable} changes?",
            "Line = expected outcome  ·  band = P10-P90  ·  dashed = where the ranking flips",
        )
        ax.legend(loc="best", labelcolor=colors.text)
    return _finish(fig, path)


__all__ = [
    "THEMES",
    "Theme",
    "compact",
    "density",
    "plot_cdf",
    "plot_distributions",
    "plot_sweep",
    "plot_tornado",
]
