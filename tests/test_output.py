"""Charts, terminal report and exports."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from rich.console import Console

import mchermes as mc
from mchermes.plotting import compact, density, get_theme
from mchermes.report import fmt


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_charts_render_and_save(coffee_result: mc.Result, tmp_path: Path, theme: str) -> None:
    r = coffee_result
    for name, draw in {
        "dist": lambda p: r.plot(p, theme=theme),
        "tornado": lambda p: r.plot_tornado("open", p, theme=theme),
        "cdf": lambda p: r.plot_cdf(p, theme=theme),
    }.items():
        path = tmp_path / f"{name}.png"
        assert isinstance(draw(str(path)), Figure)
        assert path.stat().st_size > 10_000


def test_charts_without_path_return_figures(coffee_result: mc.Result) -> None:
    fig = coffee_result.plot(options=["open"], title="Only open")
    assert fig.axes[0].get_title(loc="left") == "Only open"
    tornado = coffee_result.plot_tornado()  # best option is constant: empty tornado
    assert tornado.axes[0].texts
    assert isinstance(coffee_result.plot_cdf(), Figure)
    with pytest.raises(KeyError):
        coffee_result.plot_tornado("close")


def test_all_constant_options_plot() -> None:
    result = mc.Decision(n=10, seed=0).option("a", 0).option("b", 5).run()
    assert isinstance(result.plot(), Figure)
    assert isinstance(result.plot_cdf(), Figure)


def test_unknown_theme() -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        get_theme("neon")


def test_density_integrates_to_one() -> None:
    values = np.random.default_rng(0).normal(size=50_000)
    grid = np.linspace(-8, 8, 801)
    assert np.trapezoid(density(values, grid), grid) == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize(
    ("value", "text"),
    [(0, "0"), (950, "950"), (12_000, "12k"), (-6_634, "−6.63k"), (2.5e6, "2.5M"), (3e9, "3B")],
)
def test_compact(value: float, text: str) -> None:
    assert compact(value) == text


@pytest.mark.parametrize(
    ("value", "text"), [(0, "0"), (12_345.6, "12,346"), (12.345, "12.35"), (0.012345, "0.01235")]
)
def test_fmt(value: float, text: str) -> None:
    assert fmt(value) == text


def test_summary_prints_table(coffee_result: mc.Result) -> None:
    console = Console(record=True, width=120)
    coffee_result.summary(console)
    text = console.export_text()
    assert "★ dont_open" in text
    assert "CVaR 95%" in text
    assert "Recommendation: dont_open" in text
    assert "20,000 samples" in text


def test_summary_defaults_to_stdout(
    coffee_result: mc.Result, capsys: pytest.CaptureFixture[str]
) -> None:
    coffee_result.summary()
    assert "Recommendation" in capsys.readouterr().out


def test_to_dataframe(coffee_result: mc.Result) -> None:
    samples = coffee_result.to_dataframe()
    assert isinstance(samples, pd.DataFrame)
    assert list(samples.columns) == ["rent", "monthly_sales", "margin", "open", "dont_open"]
    assert len(samples) == 20_000
    summary = coffee_result.to_dataframe("summary")
    assert list(summary.index) == ["open", "dont_open"]
    assert summary.loc["open", "prob_loss"] == pytest.approx(coffee_result.prob_loss("open"))
    with pytest.raises(ValueError, match="kind must be"):
        coffee_result.to_dataframe("wide")


def test_to_dict_is_json_ready(coffee_result: mc.Result) -> None:
    data = coffee_result.to_dict()
    assert data["best_option"] == "dont_open"
    assert data["options"]["dont_open"]["sensitivity"] == {}
    assert isinstance(data["options"]["open"]["mean"], float)
