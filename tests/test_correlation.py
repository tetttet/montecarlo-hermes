"""Correlated inputs (Iman-Conover rank reordering)."""

import numpy as np
import pytest

import mchermes as mc
from mchermes.correlation import target_matrix
from mchermes.sensitivity import spearman


def build(rho: float, n: int = 50_000) -> mc.Decision:
    d = mc.Decision(n=n, seed=3)
    sales = d.uncertain("sales", mc.Triangular(8, 12, 18))
    margin = d.uncertain("margin", mc.Uniform(0.2, 0.4))
    d.uncertain("other", mc.Normal(0, 1))
    d.option("profit", lambda: sales * margin)
    d.correlate("sales", margin, rho)
    return d


@pytest.mark.parametrize("rho", [-0.7, -0.2, 0.3, 0.8])
def test_hits_target_rank_correlation(rho: float) -> None:
    r = build(rho).run()
    assert spearman(r.inputs["sales"], r.inputs["margin"]) == pytest.approx(rho, abs=0.01)
    assert spearman(r.inputs["sales"], r.inputs["other"]) == pytest.approx(0, abs=0.02)


def test_marginals_are_unchanged() -> None:
    independent = mc.Decision(n=5_000, seed=3)
    independent.uncertain("sales", mc.Triangular(8, 12, 18))
    independent.uncertain("margin", mc.Uniform(0.2, 0.4))
    independent.option("x", 0)
    base = independent.run()
    correlated = build(0.6, n=5_000).run()
    for name in ("sales", "margin"):
        np.testing.assert_array_equal(np.sort(base.inputs[name]), np.sort(correlated.inputs[name]))
        assert not correlated.inputs[name].flags.writeable


def test_correlation_changes_outcome_spread() -> None:
    high = build(0.8).run().stats["profit"].std
    low = build(-0.8).run().stats["profit"].std
    assert high > low * 1.5


def test_three_way_correlations_and_reproducibility() -> None:
    d = build(0.5, n=20_000)
    d.correlate("other", "sales", -0.3)
    a, b = d.run(), d.run()
    np.testing.assert_array_equal(a.inputs["sales"], b.inputs["sales"])
    assert spearman(a.inputs["other"], a.inputs["sales"]) == pytest.approx(-0.3, abs=0.02)
    assert d.correlations == {("margin", "sales"): 0.5, ("other", "sales"): -0.3}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("sales", "sales", 0.5), "with itself"),
        (("sales", "margin", 1.0), "strictly between -1 and 1"),
        (("sales", "margin", "x"), "must be a number"),
        (("sales", "nope", 0.5), "unknown variable"),
    ],
)
def test_invalid_correlations(args: tuple[str, str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build(0.1).correlate(*args)  # type: ignore[arg-type]


def test_inconsistent_correlations_rejected() -> None:
    d = build(0.9)
    d.correlate("sales", "other", 0.9).correlate("margin", "other", -0.9)
    with pytest.raises(mc.CorrelationError, match="mutually inconsistent"):
        d.run()
    with pytest.raises(mc.CorrelationError):
        target_matrix(["a", "b", "c"], {("a", "b"): 0.9, ("a", "c"): 0.9, ("b", "c"): -0.9})


def test_fixed_variable_cannot_be_correlated() -> None:
    d = mc.Decision(n=100, seed=0)
    d.uncertain("a", mc.Normal(0, 1))
    d.uncertain("b", mc.Fixed(1))
    d.option("x", 0).correlate("a", "b", 0.5)
    with pytest.raises(mc.CorrelationError, match="never varies"):
        d.run()
