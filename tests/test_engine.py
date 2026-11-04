"""Decision building, running and reproducibility."""

import numpy as np
import pytest

import mchermes as mc
from mchermes.decision import stream


def test_readme_quickstart(coffee_result: mc.Result) -> None:
    r = coffee_result
    assert r.options == ["open", "dont_open"]
    assert r.best_option() == "dont_open"
    assert 0.6 < r.prob_loss("open") < 0.85
    assert r.prob_loss("dont_open") == 0
    assert r.outcomes["open"].shape == (20_000,)
    np.testing.assert_array_equal(r.outcomes["dont_open"], 0)


def test_same_seed_identical_results(coffee: mc.Decision) -> None:
    a, b = coffee.run(), coffee.run()
    np.testing.assert_array_equal(a.outcomes["open"], b.outcomes["open"])
    c = coffee.run(seed=43)
    assert not np.array_equal(a.outcomes["open"], c.outcomes["open"])


def test_variable_streams_independent_of_order() -> None:
    def build(order: list[str]) -> mc.Result:
        d = mc.Decision(n=500, seed=3)
        handles = {name: d.uncertain(name, mc.Normal(0, 1)) for name in order}
        x = handles["x"]
        d.option("out", lambda: x)
        return d.run()

    a, b = build(["x", "y"]), build(["y", "z", "x"])
    np.testing.assert_array_equal(a.outcomes["out"], b.outcomes["out"])
    np.testing.assert_array_equal(a.inputs["x"], stream(3, "x").normal(0, 1, 500))


def test_random_seed_is_recorded_and_replayable() -> None:
    d = mc.Decision(n=100)
    v = d.uncertain("v", mc.Uniform(0, 1))
    d.option("out", lambda: v)
    first = d.run()
    assert isinstance(first.seed, int)
    replay = d.run(seed=first.seed)
    np.testing.assert_array_equal(first.outcomes["out"], replay.outcomes["out"])


def test_run_overrides_sample_size(coffee: mc.Decision) -> None:
    assert coffee.run(n=50).outcomes["open"].shape == (50,)


def test_variable_behaves_like_array() -> None:
    d = mc.Decision(n=1000, seed=1)
    x = d.uncertain("x", mc.Uniform(1, 2))
    with pytest.raises(RuntimeError, match="no samples yet"):
        _ = x.values
    d.option("ufunc", lambda: np.maximum(x, 1.5) + np.exp(0 * x))
    d.option("where", lambda: np.where(x > 1.5, 1.0, 0.0))
    d.option("handle", lambda: x)
    d.option("reverse", lambda: 10 - x)
    r = d.run()
    assert len(x) == 1000
    assert np.asarray(x).shape == (1000,)
    assert (r.outcomes["ufunc"] >= 2.5).all()
    assert set(np.unique(r.outcomes["where"])) == {0.0, 1.0}
    np.testing.assert_array_equal(r.outcomes["handle"], r.inputs["x"])
    np.testing.assert_allclose(r.outcomes["reverse"], 10 - r.inputs["x"])
    assert repr(x) == "Variable('x', Uniform(low=1.0, high=2.0))"
    assert not r.inputs["x"].flags.writeable


def test_formula_and_number_options_match_lambda(coffee: mc.Decision) -> None:
    coffee.option("formula", "monthly_sales * margin * 12 - rent * 12 - 20000")
    coffee.option("constant", 5)
    r = coffee.run()
    np.testing.assert_allclose(r.outcomes["formula"], r.outcomes["open"])
    np.testing.assert_array_equal(r.outcomes["constant"], 5)


def test_option_chaining_and_repr() -> None:
    d = mc.Decision("Q", n=10, seed=0)
    assert d.option("a", 1).option("b", 2) is d
    assert repr(d) == "Decision('Q', n=10, variables=[], options=['a', 'b'])"
    assert "best_option='b'" in repr(d.run())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 1}, "integer >= 2"),
        ({"n": 2.5}, "integer >= 2"),
        ({"n": True}, "integer >= 2"),
        ({"seed": -1}, "non-negative integer"),
        ({"seed": 1.5}, "non-negative integer"),
        ({"criterion": "vibes"}, "unknown criterion"),
    ],
)
def test_invalid_decision_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mc.Decision(**kwargs)  # type: ignore[arg-type]


def test_invalid_declarations() -> None:
    d = mc.Decision(n=10)
    d.uncertain("x", mc.Normal(0, 1))
    with pytest.raises(ValueError, match="already used"):
        d.uncertain("x", mc.Normal(0, 1))
    with pytest.raises(ValueError, match="already used"):
        d.option("x", 1)
    with pytest.raises(ValueError, match="must be an identifier"):
        d.uncertain("monthly sales", mc.Normal(0, 1))
    with pytest.raises(TypeError, match="needs a distribution"):
        d.uncertain("y", 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable, a formula string or a number"):
        d.option("z", [1, 2])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable, a formula string or a number"):
        d.option("z", True)
    with pytest.raises(mc.FormulaError, match="unknown variable"):
        d.option("z", "nope * 2")


@pytest.mark.filterwarnings("ignore:invalid value encountered in log")
def test_run_errors() -> None:
    d = mc.Decision(n=10, seed=0)
    with pytest.raises(ValueError, match="at least one option"):
        d.run()
    x = d.uncertain("x", mc.Normal(0, 1))
    d.option("bad_shape", lambda: np.ones(3))
    with pytest.raises(ValueError, match="one value per sample"):
        d.run()
    d.options.pop("bad_shape")
    d.option("nan", lambda: np.log(x - 100))
    with pytest.raises(ValueError, match="non-finite"):
        d.run()
