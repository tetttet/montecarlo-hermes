"""Value of information, comparisons and conditional results."""

import math

import numpy as np
import pytest

import mchermes as mc
from mchermes.analysis import evpi, evppi

PHI0 = 1 / math.sqrt(2 * math.pi)  # E[max(X, 0)] for X ~ N(0, 1)


@pytest.fixture(scope="module")
def gamble() -> mc.Result:
    d = mc.Decision(n=200_000, seed=4)
    x = d.uncertain("x", mc.Normal(0, 1))
    d.uncertain("noise", mc.Uniform(0, 1))
    d.uncertain("coin", mc.Bernoulli(0.5))
    d.uncertain("const", mc.Fixed(2))
    d.option("risky", lambda: x)
    d.option("safe", 0)
    return d.run()


def test_evpi_matches_analytic(gamble: mc.Result) -> None:
    assert gamble.evpi() == pytest.approx(PHI0, abs=0.01)


def test_evppi_relevant_and_irrelevant_inputs(gamble: mc.Result) -> None:
    rows = {r.variable: r for r in gamble.value_of_information()}
    assert set(rows) == {"x", "noise", "coin"}
    assert rows["x"].value == pytest.approx(PHI0, abs=0.02)
    assert rows["x"].share_of_evpi == pytest.approx(1, abs=0.05)
    assert rows["noise"].value < 0.02
    assert rows["coin"].value < 0.01
    assert gamble.value_of_information()[0].variable == "x"


def test_evpi_zero_for_dominant_option() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    assert evpi(matrix) == 0
    assert evppi(np.array([1.0, 2.0, 3.0]), matrix) == 0


def test_single_option_has_no_information_value() -> None:
    d = mc.Decision(n=1000, seed=0)
    x = d.uncertain("x", mc.Normal(0, 1))
    r = d.option("only", lambda: x).run()
    assert r.evpi() == 0
    assert all(v.share_of_evpi == 0 for v in r.value_of_information())


def test_compare(gamble: mc.Result) -> None:
    c = gamble.compare("risky", "safe")
    assert c.prob_a_better == pytest.approx(0.5, abs=0.01)
    assert c.mean_difference == pytest.approx(0, abs=0.01)
    assert c.p5 == pytest.approx(-1.645, abs=0.02)
    assert c.to_dict()["a"] == "risky"
    with pytest.raises(KeyError):
        gamble.compare("risky", "nope")


def test_prob_above(gamble: mc.Result) -> None:
    assert gamble.prob_above("risky", 1.0) == pytest.approx(0.1587, abs=0.005)


def test_given_formula(gamble: mc.Result) -> None:
    good = gamble.given("x > 1")
    assert good.n == pytest.approx(0.1587 * gamble.n, rel=0.03)
    assert good.best_option() == "risky"
    assert good.stats["risky"].min > 1
    assert good.name.endswith("| x > 1")
    both = gamble.given("coin == 1 and risky < 0")
    assert (both.inputs["coin"] == 1).all()


def test_given_mask_and_errors(gamble: mc.Result) -> None:
    mask = gamble.inputs["coin"] == 0
    assert gamble.given(mask).n == int(mask.sum())
    with pytest.raises(ValueError, match="one boolean per sample"):
        gamble.given(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="samples satisfy"):
        gamble.given("x > 100")
    with pytest.raises(mc.FormulaError):
        gamble.given("__import__('os')")
