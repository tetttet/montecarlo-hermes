"""Command-line interface: ``mchermes run`` and ``mchermes init``."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mchermes import __version__
from mchermes.decision import Decision
from mchermes.loader import ConfigError, load
from mchermes.metrics import CRITERIA
from mchermes.report import fmt, print_summary

app = typer.Typer(
    name="mchermes",
    help="🎲 Simulate your decisions before you make them.",
    no_args_is_help=True,
    add_completion=False,
)
errors = Console(stderr=True)

EXAMPLE = """\
# A decision: options, uncertain inputs and an outcome formula per option.
# Run it with:  mchermes run {path}
name: Open a second coffee shop?
samples: 100000
seed: 42
variables:
  rent: {{dist: normal, mean: 3000, std: 300}}
  monthly_sales: {{dist: triangular, low: 8000, mode: 12000, high: 18000}}
  margin: {{dist: uniform, low: 0.25, high: 0.4}}
options:
  open: "monthly_sales * margin * 12 - rent * 12 - 20000"
  dont_open: "0"
"""


def _version(value: bool) -> None:
    if value:
        typer.echo(f"mchermes {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],  # noqa: UP045 - typer needs Optional on Python 3.10
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = None,
) -> None:
    """🎲 Simulate your decisions before you make them."""


@contextmanager
def _friendly_errors(file: Path) -> Iterator[None]:
    """Turn expected failures into a red one-line message and exit code 1."""
    try:
        yield
    except FileNotFoundError:
        errors.print(f"[red]error:[/] file not found: {file}")
        raise typer.Exit(1) from None
    except (KeyError, ValueError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        errors.print(f"[red]error:[/] {message}")
        raise typer.Exit(1) from None


def _load(file: Path, criterion: str | None = None) -> Decision:
    decision = load(file)
    if criterion is not None:
        if criterion not in CRITERIA:
            raise ConfigError(f"--criterion must be one of {', '.join(CRITERIA)}")
        decision.criterion = criterion
    return decision


Samples = Annotated[
    Optional[int],  # noqa: UP045
    typer.Option("--samples", "-n", min=2, help="Override the number of samples."),
]
Seed = Annotated[
    Optional[int],  # noqa: UP045
    typer.Option("--seed", "-s", min=0, help="Override the random seed."),
]
Theme = Annotated[str, typer.Option(help="Chart theme: light or dark.")]


@app.command()
def run(
    file: Annotated[Path, typer.Argument(help="Decision file (YAML).")],
    plot: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--plot", "-p", help="Save the distributions chart to this image."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON instead of a table.")
    ] = False,
    html: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--html", help="Write a standalone interactive HTML report."),
    ] = None,
    csv: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--csv", help="Write every sample (inputs and outcomes) as CSV."),
    ] = None,
    given: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--given", "-g", help='Only futures where this holds, e.g. "rent > 3300".'),
    ] = None,
    samples: Samples = None,
    seed: Seed = None,
    criterion: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--criterion", "-c", help=f"One of: {', '.join(CRITERIA)}."),
    ] = None,
    theme: Theme = "light",
) -> None:
    """Simulate a decision file and print metrics with a recommendation."""
    with _friendly_errors(file):
        result = _load(file, criterion).run(n=samples, seed=seed)
        if given is not None:
            result = result.given(given)
        if plot is not None:
            result.plot(str(plot), theme=theme)
        if html is not None:
            result.to_html(html)
        if csv is not None:
            result.to_csv(csv)
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    console = Console()
    print_summary(result, console)
    for label, path in (("Chart", plot), ("Report", html), ("Samples", csv)):
        if path is not None:
            console.print(f"[dim]{label} saved to {path}[/]")


@app.command()
def sweep(
    file: Annotated[Path, typer.Argument(help="Decision file (YAML).")],
    variable: Annotated[str, typer.Argument(help="Input to vary.")],
    start: Annotated[float, typer.Argument(help="First value.")],
    stop: Annotated[float, typer.Argument(help="Last value.")],
    steps: Annotated[int, typer.Option("--steps", "-k", min=2, help="Number of values.")] = 11,
    plot: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--plot", "-p", help="Save the sweep chart to this image."),
    ] = None,
    samples: Samples = None,
    seed: Seed = None,
    theme: Theme = "light",
) -> None:
    """What if VARIABLE were fixed at each value from START to STOP? Finds breakevens."""
    with _friendly_errors(file):
        decision = _load(file)
        if seed is not None:
            decision.seed = seed
        result = decision.sweep(variable, np.linspace(start, stop, steps), n=samples)
        if plot is not None:
            result.plot(str(plot), theme=theme)
    console = Console()
    table = Table(
        title=f"[bold]{escape(decision.name)}[/]: sweep of {variable}", title_justify="left"
    )
    table.add_column(variable, justify="right")
    for option in result.options:
        table.add_column(f"mean {option}", justify="right")
    table.add_column("Best", style="bold green")
    for row in result.to_rows():
        means = [fmt(row[f"mean_{o}"]) for o in result.options]
        table.add_row(fmt(row[variable]), *means, row["best_option"])
    console.print(table)
    for i, a in enumerate(result.options):
        for b in result.options[i + 1 :]:
            point = result.breakeven(a, b)
            if point is not None:
                console.print(f"Breakeven {a} vs {b}: {variable} ≈ [bold]{fmt(point)}[/]")
    if plot is not None:
        console.print(f"[dim]Chart saved to {plot}[/]")


@app.command()
def validate(file: Annotated[Path, typer.Argument(help="Decision file (YAML).")]) -> None:
    """Check a decision file without running the full simulation."""
    with _friendly_errors(file):
        decision = _load(file)
        decision.run(n=100)
    typer.echo(
        f"OK: {file} ({len(decision.variables)} variables, {len(decision.options)} options, "
        f"{len(decision.correlations)} correlations)"
    )


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the example.")] = Path(
        "decision.yaml"
    ),
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite an existing file.")] = (
        False
    ),
) -> None:
    """Write an example decision file to get started."""
    if path.exists() and not force:
        errors.print(f"[red]error:[/] {path} already exists (use --force to overwrite)")
        raise typer.Exit(1)
    path.write_text(EXAMPLE.format(path=path), encoding="utf-8")
    typer.echo(f"Wrote {path}. Next: mchermes run {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
