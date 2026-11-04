"""Parameter sweeps and breakeven."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mchermes as mc


@pytest.fixture
def pricing() -> mc.Decision:
    d = mc.Decision("Price", n=5_000, seed=1)
    price = d.uncertain("price", mc.Uniform(8, 12))
    demand = d.uncertain("demand", mc.Normal(100, 10))
    d.uncertain("other", mc.Normal(0, 1))
    d.option("sell", lambda: (price - 10) * demand)
    d.option("hold", 0)
    d.correlate("demand", "price", 0.3)
    return d


def test_sweep_finds_breakeven(pricing: mc.Decision) -> None:
    sweep = pricing.sweep("price", np.linspace(8, 12, 9))
    assert sweep.options == ["sell", "hold"]
    assert sweep.breakeven("sell", "hold") == pytest.approx(10, abs=1e-6)
    assert sweep.best_options()[0] == "hold"
    assert sweep.best_options()[-1] == "sell"
    np.testing.assert_allclose(sweep.means("hold"), 0)
    assert (sweep.percentiles("sell", 90) >= sweep.means("sell")).all()
    assert "9 values" in repr(sweep)


def test_sweep_restores_decision(pricing: mc.Decision) -> None:
    before = pricing.run()
    pricing.sweep("price", [9, 11])
    assert pricing.variables["price"].distribution == mc.Uniform(8, 12)
    assert pricing.correlations == {("demand", "price"): 0.3}
    np.testing.assert_array_equal(before.outcomes["sell"], pricing.run().outcomes["sell"])


def test_breakeven_interpolates_and_can_be_absent(pricing: mc.Decision) -> None:
    sweep = pricing.sweep(pricing.variables["price"], [9, 10.5])
    assert sweep.breakeven("sell", "hold") == pytest.approx(10, abs=0.05)
    assert pricing.sweep("price", [11, 12]).breakeven("sell", "hold") is None


def test_sweep_exports(pricing: mc.Decision, tmp_path: Path) -> None:
    sweep = pricing.sweep("price", [9, 10, 11], n=500)
    rows = sweep.to_rows()
    assert rows[0] == {"price": 9.0, "mean_sell": rows[0]["mean_sell"], "mean_hold": 0.0,
                       "best_option": "hold"}  # fmt: skip
    frame = sweep.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == [9.0, 10.0, 11.0]
    path = tmp_path / "sweep.png"
    sweep.plot(str(path), theme="dark")
    assert path.stat().st_size > 10_000


def test_sweep_errors(pricing: mc.Decision) -> None:
    with pytest.raises(ValueError, match="unknown variable"):
        pricing.sweep("nope", [1])
    with pytest.raises(ValueError, match="at least one value"):
        pricing.sweep("price", [])


def test_sweep_without_seed_uses_common_random_numbers() -> None:
    d = mc.Decision(n=2_000)
    x = d.uncertain("x", mc.Normal(0, 1))
    k = d.uncertain("k", mc.Uniform(0, 1))
    d.option("a", lambda: x + k)
    sweep = d.sweep("k", [0, 1])
    diff = sweep.results[1].outcomes["a"] - sweep.results[0].outcomes["a"]
    np.testing.assert_allclose(diff, 1)
