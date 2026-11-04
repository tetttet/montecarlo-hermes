"""Distributions: shapes, moments, validation and reproducibility."""

import numpy as np
import pytest

import mchermes as mc

ALL = [
    mc.Normal(10, 2),
    mc.LogNormal(5, 3),
    mc.Uniform(-1, 3),
    mc.Triangular(1, 2, 6),
    mc.PERT(1, 2, 6),
    mc.Beta(2, 5, low=10, high=20),
    mc.Bernoulli(0.3),
    mc.Discrete([1, 2, 10], [0.2, 0.5, 0.3]),
    mc.Fixed(4.5),
]


@pytest.mark.parametrize("dist", ALL, ids=repr)
def test_shape_dtype_and_moments(dist: mc.Distribution) -> None:
    values = dist.sample(np.random.default_rng(0), 200_000)
    assert values.shape == (200_000,)
    assert values.dtype == np.float64
    assert values.mean() == pytest.approx(dist.mean, rel=0.01, abs=0.01)
    assert values.std() == pytest.approx(dist.std, rel=0.02, abs=0.01)


@pytest.mark.parametrize("dist", ALL, ids=repr)
def test_same_seed_same_draws(dist: mc.Distribution) -> None:
    a = dist.sample(np.random.default_rng(7), 1000)
    b = dist.sample(np.random.default_rng(7), 1000)
    np.testing.assert_array_equal(a, b)


def test_supports() -> None:
    rng = np.random.default_rng(1)
    assert set(np.unique(mc.Bernoulli(0.5).sample(rng, 1000))) == {0.0, 1.0}
    assert set(np.unique(mc.Discrete([3, 7], [0.5, 0.5]).sample(rng, 1000))) == {3.0, 7.0}
    tri = mc.Triangular(1, 2, 6).sample(rng, 10_000)
    assert tri.min() >= 1
    assert tri.max() <= 6
    beta = mc.Beta(2, 2, low=10, high=20).sample(rng, 10_000)
    assert beta.min() >= 10
    assert beta.max() <= 20
    assert (mc.LogNormal(1, 1).sample(rng, 1000) > 0).all()


def test_lognormal_from_log_params() -> None:
    dist = mc.LogNormal.from_log_params(0.0, 0.5)
    assert dist.mu == pytest.approx(0.0)
    assert dist.sigma == pytest.approx(0.5)


def test_pert_is_beta_with_mode_weight() -> None:
    pert = mc.PERT(0, 5, 10)
    assert pert.alpha == pytest.approx(3)
    assert pert.beta == pytest.approx(3)
    assert pert.mean == pytest.approx(5)


def test_repr_and_equality() -> None:
    assert repr(mc.Normal(1, 2)) == "Normal(mean=1.0, std=2.0)"
    assert repr(mc.PERT(1, 2, 3)) == "PERT(low=1.0, mode=2.0, high=3.0, lamb=4.0)"
    assert mc.Uniform(0, 1) == mc.Uniform(0, 1)
    assert mc.Uniform(0, 1) != mc.Uniform(0, 2)
    assert len({mc.Fixed(1), mc.Fixed(1)}) == 1
    assert "mean=5.0" in repr(mc.LogNormal(5, 1))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: mc.Normal(0, 0), "std must be > 0"),
        (lambda: mc.Normal(0, -1), "std must be > 0"),
        (lambda: mc.Normal(float("nan"), 1), "must be finite"),
        (lambda: mc.Normal("a", 1), "must be a number"),
        (lambda: mc.LogNormal(-1, 1), "mean must be > 0"),
        (lambda: mc.LogNormal.from_log_params(0, 0), "sigma must be > 0"),
        (lambda: mc.Uniform(2, 1), "low < high"),
        (lambda: mc.Triangular(1, 5, 3), "low <= mode <= high"),
        (lambda: mc.Triangular(2, 2, 2), "low < high"),
        (lambda: mc.PERT(0, 11, 10), "low <= mode <= high"),
        (lambda: mc.PERT(0, 1, 10, lamb=0), "lamb must be > 0"),
        (lambda: mc.Beta(0, 1), "alpha must be > 0"),
        (lambda: mc.Beta(1, 1, low=1, high=0), "low < high"),
        (lambda: mc.Bernoulli(1.5), r"\[0, 1\]"),
        (lambda: mc.Discrete([], []), "at least one value"),
        (lambda: mc.Discrete([1, 2], [1.0]), "one probability per value"),
        (lambda: mc.Discrete([1, 2], [1.2, -0.2]), ">= 0"),
        (lambda: mc.Discrete([1, 2], [0.5, 0.6]), "sum to 1"),
        (lambda: mc.Discrete("ab", [0.5, 0.5]), "lists of numbers"),
        (lambda: mc.Fixed(float("inf")), "must be finite"),
    ],
)
def test_invalid_parameters(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
