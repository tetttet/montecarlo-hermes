"""Command-line interface: ``mchermes run`` and ``mchermes init``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from mchermes import __version__
from mchermes.loader import ConfigError, load
from mchermes.metrics import CRITERIA
from mchermes.report import print_summary

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
    samples: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option("--samples", "-n", min=2, help="Override the number of samples."),
    ] = None,
    seed: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option("--seed", "-s", min=0, help="Override the random seed."),
    ] = None,
    criterion: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--criterion", "-c", help=f"One of: {', '.join(CRITERIA)}."),
    ] = None,
    theme: Annotated[str, typer.Option(help="Chart theme: light or dark.")] = "light",
) -> None:
    """Simulate a decision file and print metrics with a recommendation."""
    try:
        decision = load(file)
        if criterion is not None:
            if criterion not in CRITERIA:
                raise ConfigError(f"--criterion must be one of {', '.join(CRITERIA)}")
            decision.criterion = criterion
        result = decision.run(n=samples, seed=seed)
        if plot is not None:
            result.plot(str(plot), theme=theme)
    except FileNotFoundError:
        errors.print(f"[red]error:[/] file not found: {file}")
        raise typer.Exit(1) from None
    except ValueError as error:
        errors.print(f"[red]error:[/] {error}")
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    console = Console()
    print_summary(result, console)
    if plot is not None:
        console.print(f"[dim]Chart saved to {plot}[/]")


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
