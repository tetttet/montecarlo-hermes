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
    mc.Exponential(3),
    mc.Gamma(2, 3),
    mc.Gamma.from_mean_std(10, 4),
    mc.Weibull(1.5, 10),
    mc.StudentT(5, loc=1, scale=2),
    mc.Poisson(4),
    mc.Binomial(20, 0.3),
    mc.Empirical([1, 2, 3, 10]),
    mc.Normal(0, 1).clip(0),
    mc.Normal(0, 1).clip(-1, 1),
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
        (lambda: mc.Exponential(0), "mean must be > 0"),
        (lambda: mc.Gamma(0, 1), "shape must be > 0"),
        (lambda: mc.Gamma.from_mean_std(1, 0), "std must be > 0"),
        (lambda: mc.Weibull(1, -1), "scale must be > 0"),
        (lambda: mc.StudentT(0), "df must be > 0"),
        (lambda: mc.Poisson(0), "lam must be > 0"),
        (lambda: mc.Binomial(2.5, 0.5), "non-negative integer"),
        (lambda: mc.Binomial(-1, 0.5), "non-negative integer"),
        (lambda: mc.Binomial(10, 2), r"\[0, 1\]"),
        (lambda: mc.Empirical([]), "at least one observation"),
        (lambda: mc.Empirical("abc"), "list of numbers"),
        (lambda: mc.Empirical([1, "x"]), "list of numbers"),
        (lambda: mc.Empirical([1, float("nan")]), "must be finite"),
        (lambda: mc.Normal(0, 1).clip(), "low, high or both"),
        (lambda: mc.Normal(0, 1).clip(2, 1), "low < high"),
    ],
)
def test_invalid_parameters(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_draw_is_seeded() -> None:
    a = mc.PERT(1, 2, 6).draw(5, seed=0)
    np.testing.assert_array_equal(a, mc.PERT(1, 2, 6).draw(5, seed=0))
    assert a.shape == (5,)


def test_counts_are_integers() -> None:
    values = mc.Poisson(3).draw(1000, seed=1)
    np.testing.assert_array_equal(values, np.round(values))
    assert mc.Binomial(10, 0.5).draw(1000, seed=1).max() <= 10


def test_empirical_resamples_data() -> None:
    data = [3.0, 5.0, 8.0]
    assert set(np.unique(mc.Empirical(data).draw(1000, seed=0))) == set(data)
    assert mc.Empirical(np.array(data)) == mc.Empirical(data)
    assert mc.Empirical(data) != mc.Empirical([1.0])
    assert len({mc.Empirical(data), mc.Empirical(data)}) == 1
    assert repr(mc.Empirical(data)) == "Empirical(n=3, mean=5.333333333333)"


def test_clip_bounds_values() -> None:
    values = mc.Normal(0, 1).clip(-1, 1).draw(10_000, seed=0)
    assert values.min() == -1
    assert values.max() == 1
    with pytest.raises(TypeError, match="distribution to clip"):
        mc.Clipped(5, 0, 1)  # type: ignore[arg-type]


def test_student_t_moments_for_low_df() -> None:
    assert np.isnan(mc.StudentT(1).mean)
    assert np.isnan(mc.StudentT(1).std)
    assert mc.StudentT(1.5).std == float("inf")
