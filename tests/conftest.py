"""Shared fixtures."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

import mchermes as mc

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def coffee() -> mc.Decision:
    """The README coffee-shop decision with a small sample size."""
    d = mc.Decision("Open a second coffee shop?", n=20_000, seed=42)
    rent = d.uncertain("rent", mc.Normal(3000, 300))
    sales = d.uncertain("monthly_sales", mc.Triangular(8000, 12000, 18000))
    margin = d.uncertain("margin", mc.Uniform(0.25, 0.4))
    d.option("open", lambda: sales * margin * 12 - rent * 12 - 20_000)
    d.option("dont_open", lambda: 0)
    return d


@pytest.fixture
def coffee_result(coffee: mc.Decision) -> mc.Result:
    """Result of the coffee-shop decision."""
    return coffee.run()
