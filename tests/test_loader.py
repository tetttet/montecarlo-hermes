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
        ({"dist": "gamma"}, "'dist' must be one of"),
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
