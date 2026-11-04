"""The mchermes command line, through typer's CliRunner."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mchermes import __version__
from mchermes.cli import app

from .conftest import EXAMPLES

runner = CliRunner()
COFFEE = str(EXAMPLES / "coffee_shop.yaml")


def test_run_prints_table_and_recommendation() -> None:
    result = runner.invoke(app, ["run", COFFEE, "-n", "2000"])
    assert result.exit_code == 0, result.output
    assert "Open a second coffee shop?" in result.output
    assert "P(loss)" in result.output
    assert "Recommendation: dont_open" in result.output


def test_run_json() -> None:
    result = runner.invoke(app, ["run", COFFEE, "--json", "-n", "2000", "--seed", "1"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["best_option"] == "dont_open"
    assert data["samples"] == 2000
    assert data["seed"] == 1
    assert set(data["options"]) == {"open", "dont_open"}
    assert data["options"]["open"]["sensitivity"]["monthly_sales"] > 0


def test_run_criterion_override() -> None:
    result = runner.invoke(
        app, ["run", str(EXAMPLES / "job_offer.yaml"), "-n", "20000", "-c", "risk_averse", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["criterion"] == "risk_averse"
    assert data["best_option"] == "stay"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_run_plot(tmp_path: Path, theme: str) -> None:
    out = tmp_path / "chart.png"
    result = runner.invoke(app, ["run", COFFEE, "-n", "2000", "--plot", str(out), "--theme", theme])
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 10_000
    assert "Chart saved" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["run", "missing.yaml"], "file not found"),
        (["run", COFFEE, "-c", "vibes"], "--criterion must be one of"),
        (["run", COFFEE, "-n", "100", "--plot", "x.png", "--theme", "neon"], "unknown theme"),
    ],
)
def test_run_errors(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert message in result.output


def test_run_reports_config_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("options:\n  go: \"__import__('os')\"\n", encoding="utf-8")
    result = runner.invoke(app, ["run", str(path)])
    assert result.exit_code == 1
    assert "not allowed" in result.output


def test_init_writes_runnable_example(tmp_path: Path) -> None:
    path = tmp_path / "decision.yaml"
    result = runner.invoke(app, ["init", str(path)])
    assert result.exit_code == 0, result.output
    assert path.exists()
    assert runner.invoke(app, ["run", str(path), "-n", "500"]).exit_code == 0
    again = runner.invoke(app, ["init", str(path)])
    assert again.exit_code == 1
    assert "already exists" in again.output
    assert runner.invoke(app, ["init", str(path), "--force"]).exit_code == 0


def test_version_and_help() -> None:
    assert __version__ in runner.invoke(app, ["--version"]).output
    help_text = runner.invoke(app, ["--help"]).output
    assert "run" in help_text
    assert "init" in help_text


def test_run_html_csv_and_given(tmp_path: Path) -> None:
    html, csv = tmp_path / "r.html", tmp_path / "s.csv"
    result = runner.invoke(
        app,
        ["run", COFFEE, "-n", "2000", "--html", str(html), "--csv", str(csv), "-g", "rent > 3300"],
    )
    assert result.exit_code == 0, result.output
    assert "| rent > 3300" in result.output
    assert "Report saved" in result.output
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert csv.read_text(encoding="utf-8").startswith("rent,monthly_sales,margin")


def test_run_given_errors() -> None:
    result = runner.invoke(app, ["run", COFFEE, "-n", "500", "-g", "nope > 1"])
    assert result.exit_code == 1
    assert "unknown variable" in result.output


def test_sweep_command(tmp_path: Path) -> None:
    out = tmp_path / "sweep.png"
    args = ["sweep", COFFEE, "monthly_sales", "8000", "18000", "-k", "6", "-n", "5000"]
    result = runner.invoke(app, [*args, "--seed", "3", "--plot", str(out)])
    assert result.exit_code == 0, result.output
    assert "mean open" in result.output
    assert "Breakeven open vs dont_open: monthly_sales ≈ 14," in result.output
    assert out.stat().st_size > 10_000


def test_sweep_command_errors() -> None:
    result = runner.invoke(app, ["sweep", COFFEE, "nope", "1", "2"])
    assert result.exit_code == 1
    assert "unknown variable" in result.output


def test_validate_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(EXAMPLES / "import_business.yaml")])
    assert result.exit_code == 0
    assert "OK:" in result.output
    assert "11 variables, 2 options, 1 correlations" in result.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("options: {go: 'missing * 2'}\n", encoding="utf-8")
    failed = runner.invoke(app, ["validate", str(bad)])
    assert failed.exit_code == 1
    assert "unknown variable" in failed.output
