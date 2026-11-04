"""Metrics against known analytic values."""

import math

import numpy as np
import pytest

import mchermes as mc
from mchermes.metrics import best_share, cvar, expected_regret, rank_by, summarize

N = 400_000


@pytest.fixture(scope="module")
def normal_vs_zero() -> mc.Result:
    d = mc.Decision("N(0,1) vs 0", n=N, seed=11)
    x = d.uncertain("x", mc.Normal(0, 1))
    d.option("risky", lambda: x)
    d.option("safe", 0)
    return d.run()


def test_normal_moments_and_percentiles(normal_vs_zero: mc.Result) -> None:
    s = normal_vs_zero.stats["risky"]
    assert s.mean == pytest.approx(0, abs=0.01)
    assert s.std == pytest.approx(1, abs=0.01)
    assert s.median == pytest.approx(0, abs=0.01)
    assert s.p50 == s.median
    for value, z in [(s.p5, -1.6449), (s.p10, -1.2816), (s.p90, 1.2816), (s.p95, 1.6449)]:
        assert value == pytest.approx(z, abs=0.02)
    assert s.min < -4
    assert s.max > 4


def test_normal_risk_metrics(normal_vs_zero: mc.Result) -> None:
    s = normal_vs_zero.stats["risky"]
    assert s.prob_loss == pytest.approx(0.5, abs=0.005)
    assert s.var_95 == s.p5
    # Expected shortfall of N(0,1) at 95%: -phi(1.645) / 0.05
    assert s.cvar_95 == pytest.approx(-2.0627, abs=0.02)


def test_prob_best_and_regret(normal_vs_zero: mc.Result) -> None:
    risky, safe = normal_vs_zero.stats["risky"], normal_vs_zero.stats["safe"]
    assert risky.prob_best == pytest.approx(0.5, abs=0.005)
    assert risky.prob_best + safe.prob_best == pytest.approx(1)
    # E[max(0, X) - X] = E[max(0, X)] = 1 / sqrt(2 pi) for both options
    assert risky.expected_regret == pytest.approx(1 / math.sqrt(2 * math.pi), abs=0.01)
    assert safe.expected_regret == pytest.approx(1 / math.sqrt(2 * math.pi), abs=0.01)


def test_uniform_percentiles() -> None:
    d = mc.Decision(n=N, seed=5)
    u = d.uncertain("u", mc.Uniform(0, 100))
    d.option("out", lambda: u)
    s = d.run().stats["out"]
    assert (s.p5, s.p10, s.p90, s.p95) == pytest.approx((5, 10, 90, 95), abs=0.3)
    assert s.std == pytest.approx(100 / math.sqrt(12), rel=0.01)
    assert s.cvar_95 == pytest.approx(2.5, abs=0.1)


def test_ties_split_best_share() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 4.0]])
    np.testing.assert_allclose(best_share(matrix), [0.5, 0.5])
    np.testing.assert_allclose(expected_regret(matrix), [1 / 3, 1 / 3])


def test_cvar_edge_cases() -> None:
    assert cvar(np.arange(100.0), 0.95) == pytest.approx(2.0)
    assert cvar(np.array([5.0]), 0.95) == 5.0
    with pytest.raises(ValueError, match="level"):
        cvar(np.arange(10.0), 1.0)


def test_single_sample_std_is_zero() -> None:
    assert summarize({"a": np.array([3.0])})["a"].std == 0.0


def test_criteria_can_disagree() -> None:
    d = mc.Decision(n=100_000, seed=1)
    gamble = d.uncertain("payoff", mc.Normal(110, 60))
    d.option("gamble", lambda: gamble)
    d.option("sure", 100)
    r = d.run()
    assert r.best_option() == "gamble"
    assert r.best_option("expected_value") == "gamble"
    assert r.best_option("risk_averse") == "sure"
    assert r.best_option("min_regret") == "gamble"
    with pytest.raises(ValueError, match="unknown criterion"):
        r.best_option("vibes")


def test_decision_criterion_is_default() -> None:
    d = mc.Decision(n=10_000, seed=1, criterion="risk_averse")
    g = d.uncertain("g", mc.Normal(110, 60))
    d.option("gamble", lambda: g).option("sure", 100)
    assert d.run().best_option() == "sure"


def test_rank_by_is_stable_on_ties() -> None:
    stats = summarize({"first": np.zeros(5), "second": np.zeros(5)})
    assert rank_by(stats, "expected_value") == ["first", "second"]


def test_result_helpers(coffee_result: mc.Result) -> None:
    r = coffee_result
    assert r.prob_loss("open", threshold=-1e9) == 0
    assert r.percentile("open", 50) == pytest.approx(r.stats["open"].median)
    assert r.percentile("open", [5, 95]).shape == (2,)
    with pytest.raises(KeyError, match="unknown option"):
        r.prob_loss("close")
    stats = r.stats["open"].to_dict()
    assert set(stats) >= {"mean", "p5", "cvar_95", "prob_best", "expected_regret"}
