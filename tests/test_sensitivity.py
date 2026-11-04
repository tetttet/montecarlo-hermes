"""Spearman tornado sensitivity."""

import math

import numpy as np
import pytest

import mchermes as mc
from mchermes.sensitivity import rank, spearman, tornado


def test_rank_averages_ties() -> None:
    np.testing.assert_array_equal(rank(np.array([10.0, 20.0, 10.0, 5.0])), [2.5, 4, 2.5, 1])


def test_spearman_monotone_and_constant() -> None:
    x = np.linspace(0, 1, 100)
    assert spearman(x, np.exp(x)) == pytest.approx(1)
    assert spearman(x, -(x**3)) == pytest.approx(-1)
    assert spearman(x, np.ones(100)) == 0.0


def test_spearman_matches_bivariate_normal_formula() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200_000), rng.normal(size=200_000)
    expected = 6 / math.pi * math.asin(1 / math.sqrt(2) / 2)
    assert spearman(a, a + b) == pytest.approx(expected, abs=0.01)


def test_tornado_orders_by_impact_and_skips_constants(coffee_result: mc.Result) -> None:
    rows = coffee_result.sensitivity("open")
    assert [r.variable for r in rows] == ["monthly_sales", "margin", "rent"]
    assert rows[0].spearman > 0
    assert rows[-1].spearman < 0
    assert rows[-1].impact == abs(rows[-1].spearman)
    assert tornado({"fixed": np.ones(10)}, np.arange(10.0)) == []
