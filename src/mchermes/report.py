"""Terminal rendering of results with rich."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from mchermes.result import Result

# (label, OptionStats attribute, is a probability)
ROWS = (
    ("Mean", "mean", False),
    ("Median", "median", False),
    ("Std dev", "std", False),
    ("P5", "p5", False),
    ("P10", "p10", False),
    ("P90", "p90", False),
    ("P95", "p95", False),
    ("Min", "min", False),
    ("Max", "max", False),
    ("P(loss)", "prob_loss", True),
    ("VaR 95%", "var_95", False),
    ("CVaR 95%", "cvar_95", False),
    ("P(best)", "prob_best", True),
    ("Exp. regret", "expected_regret", False),
)


def fmt(value: float) -> str:
    """Readable number: thousands separators, sensible precision."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def summary_table(result: Result) -> Table:
    """Rich table with one column per option; the best option is starred."""
    best = result.best_option()
    table = Table(header_style="bold")
    table.add_column("Metric", style="dim")
    for option in result.options:
        label = f"★ {option}" if option == best else option
        table.add_column(label, justify="right", style="bold green" if option == best else None)
    for label, attribute, is_probability in ROWS:
        cells = []
        for option in result.options:
            value = getattr(result.stats[option], attribute)
            cells.append(f"{value:.1%}" if is_probability else fmt(value))
        table.add_row(label, *cells)
    return table


def recommendation(result: Result) -> Text:
    """One-line verdict for the best option."""
    best = result.best_option()
    stats = result.stats[best]
    text = Text("Recommendation: ", style="bold")
    text.append(best, style="bold green")
    text.append(
        f"  (expected {fmt(stats.mean)}, P(loss) {stats.prob_loss:.1%}, "
        f"best in {stats.prob_best:.1%} of scenarios)"
    )
    return text


def print_summary(result: Result, console: Console | None = None) -> None:
    """Print the summary table and the recommendation."""
    console = console or Console()
    console.print(Text(result.name, style="bold"))
    console.print(summary_table(result))
    console.print(
        f"[dim]{result.n:,} samples · seed {result.seed} · criterion {result.criterion}[/]"
    )
    console.print(recommendation(result))
    if len(result.options) > 1:
        rows = result.value_of_information()[:3]
        drivers = ", ".join(f"{r.variable} {fmt(r.value)}" for r in rows if r.value > 0)
        line = f"Value of perfect information: {fmt(result.evpi())}"
        console.print(
            f"[dim]{line}" + (f" · worth learning first: {drivers}" if drivers else "") + "[/]"
        )
