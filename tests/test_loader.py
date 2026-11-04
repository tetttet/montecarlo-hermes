"""YAML decision files."""

from pathlib import Path

import numpy as np
import pytest

import mchermes as mc
from mchermes.loader import make_distribution

from .conftest import EXAMPLES


def test_coffee_yaml_matches_python(coffee: mc.Decision) -> None:
    yaml_result = mc.load(EXAMPLES / "coffee_shop.yaml").run(n=20_000)
    py_result = coffee.run()
    np.testing.assert_allclose(yaml_result.outcomes["open"], py_result.outcomes["open"])
    assert yaml_result.name == "Open a second coffee shop?"
    assert yaml_result.seed == 42


@pytest.mark.parametrize("name", ["coffee_shop", "job_offer", "import_business"])
def test_examples_load_and_run(name: str) -> None:
    result = mc.load(EXAMPLES / f"{name}.yaml").run(n=2_000)
    assert len(result.options) == 2
    assert result.best_option() in result.options


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (5, mc.Fixed(5)),
        (2.5, mc.Fixed(2.5)),
        ({"dist": "Normal", "mean": 1, "std": 2}, mc.Normal(1, 2)),
        ({"dist": "lognormal", "mean": 1, "std": 2}, mc.LogNormal(1, 2)),
        ({"dist": "uniform", "low": 0, "high": 1}, mc.Uniform(0, 1)),
        ({"dist": "triangular", "low": 0, "mode": 1, "high": 2}, mc.Triangular(0, 1, 2)),
        ({"dist": "pert", "low": 0, "mode": 1, "high": 2}, mc.PERT(0, 1, 2)),
        ({"dist": "beta", "alpha": 2, "beta": 3}, mc.Beta(2, 3)),
        ({"dist": "bernoulli", "p": 0.2}, mc.Bernoulli(0.2)),
        (
            {"dist": "discrete", "values": [1, 2], "probabilities": [0.5, 0.5]},
            mc.Discrete([1, 2], [0.5, 0.5]),
        ),
        ({"dist": "fixed", "value": 3}, mc.Fixed(3)),
        ({"dist": "exponential", "mean": 2}, mc.Exponential(2)),
        ({"dist": "gamma", "shape": 2, "scale": 3}, mc.Gamma(2, 3)),
        ({"dist": "weibull", "shape": 2, "scale": 3}, mc.Weibull(2, 3)),
        ({"dist": "student_t", "df": 4}, mc.StudentT(4)),
        ({"dist": "poisson", "lam": 3}, mc.Poisson(3)),
        ({"dist": "binomial", "trials": 10, "p": 0.5}, mc.Binomial(10, 0.5)),
        ({"dist": "empirical", "data": [1, 2, 3]}, mc.Empirical([1, 2, 3])),
        (
            {"dist": "normal", "mean": 0, "std": 1, "clip_low": 0},
            mc.Normal(0, 1).clip(0),
        ),
    ],
)
def test_make_distribution(spec: object, expected: mc.Distribution) -> None:
    assert make_distribution("x", spec) == expected


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (True, "expected a number or a mapping"),
        ("normal", "expected a number or a mapping"),
        ({"mean": 1}, "'dist' must be one of"),
        ({"dist": "cauchy"}, "'dist' must be one of"),
        ({"dist": "gamma"}, "missing scale, shape"),
        ({"dist": "normal", "mean": 1}, "missing std"),
        ({"dist": "normal", "mean": 1, "std": 1, "skew": 2}, "unknown skew"),
        ({"dist": "normal", "mean": 1, "std": -1}, "std must be > 0"),
    ],
)
def test_bad_distribution(spec: object, message: str) -> None:
    with pytest.raises(mc.ConfigError, match=rf"variables\.x: .*{message}"):
        make_distribution("x", spec)


BASE = {"variables": {"a": 1}, "options": {"go": "a * 2"}}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "mapping at the top level"),
        ({**BASE, "colour": "red"}, "unknown top-level keys: colour"),
        ({"variables": [1], "options": {"x": "1"}}, "'variables' must be a mapping"),
        ({"variables": {}}, "'options' must be a non-empty"),
        ({"variables": {}, "options": {}}, "'options' must be a non-empty"),
        ({**BASE, "samples": 1}, "integer >= 2"),
        ({"variables": {"a": {"dist": "nope"}}, "options": {"x": "1"}}, r"variables\.a: 'dist'"),
        ({**BASE, "criterion": "vibes"}, "unknown criterion"),
        ({"variables": {"a b": 1}, "options": {"x": "1"}}, "must be an identifier"),
        ({"variables": {"a": 1}, "options": {"go": [1]}}, r"options\.go: expected a formula"),
        ({"variables": {"a": 1}, "options": {"go": "b"}}, r"options\.go: unknown variable"),
        ({"variables": {"a": 1}, "options": {"a": "1"}}, r"options\.a: name 'a' is already used"),
        (
            {"variables": {"a": 1}, "options": {"go": "__import__('os')"}},
            "'__import__' is not allowed",
        ),
        ({**BASE, "correlations": {"a": 1}}, "'correlations' must be a list"),
        ({**BASE, "correlations": [["a", 0.5]]}, r"correlations\[0\]: expected \["),
        ({**BASE, "correlations": [["a", "zz", 0.5]]}, r"correlations\[0\]: unknown variable"),
        (
            {
                "variables": {
                    "a": {"dist": "normal", "mean": 0, "std": 1, "clip_low": 1, "clip_high": 0}
                },
                "options": {"x": "a"},
            },
            "clip needs low < high",
        ),
    ],
)
def test_bad_config(data: object, message: str) -> None:
    with pytest.raises(mc.ConfigError, match=message):
        mc.from_dict(data)  # type: ignore[arg-type]


def test_from_dict_defaults_and_numbers() -> None:
    d = mc.from_dict({"options": {"flat": 3}})
    assert d.name == "Decision"
    assert d.n == 100_000
    assert d.criterion == "expected_value"
    np.testing.assert_array_equal(d.run(n=5).outcomes["flat"], 3)


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("options: [unclosed", encoding="utf-8")
    with pytest.raises(mc.ConfigError, match="invalid YAML"):
        mc.load(path)


def test_yaml_correlations(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "variables:\n"
        "  a: {dist: normal, mean: 0, std: 1}\n"
        "  b: {dist: uniform, low: 0, high: 1}\n"
        "correlations:\n"
        "  - [a, b, 0.6]\n"
        "options:\n"
        "  go: a + b\n",
        encoding="utf-8",
    )
    decision = mc.Decision.from_yaml(path)
    assert decision.correlations == {("a", "b"): 0.6}
    result = decision.run(n=20_000, seed=1)
    from mchermes.sensitivity import spearman

    assert spearman(result.inputs["a"], result.inputs["b"]) == pytest.approx(0.6, abs=0.02)


def test_decision_from_dict_alias() -> None:
    decision = mc.Decision.from_dict({"variables": {"a": 2}, "options": {"go": "a * 3"}})
    np.testing.assert_array_equal(decision.run(n=3).outcomes["go"], 6)
