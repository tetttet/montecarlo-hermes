"""Build a :class:`~mchermes.Decision` from YAML (or an equivalent ``dict``)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from mchermes.decision import Decision
from mchermes.distributions import (
    PERT,
    Bernoulli,
    Beta,
    Binomial,
    Discrete,
    Distribution,
    Empirical,
    Exponential,
    Fixed,
    Gamma,
    LogNormal,
    Normal,
    Poisson,
    StudentT,
    Triangular,
    Uniform,
    Weibull,
)
from mchermes.formula import FormulaError

DISTRIBUTIONS: dict[str, Callable[..., Distribution]] = {
    "normal": Normal,
    "lognormal": LogNormal,
    "uniform": Uniform,
    "triangular": Triangular,
    "pert": PERT,
    "beta": Beta,
    "bernoulli": Bernoulli,
    "discrete": Discrete,
    "fixed": Fixed,
    "exponential": Exponential,
    "gamma": Gamma,
    "weibull": Weibull,
    "student_t": StudentT,
    "poisson": Poisson,
    "binomial": Binomial,
    "empirical": Empirical,
}
TOP_LEVEL = {
    "name",
    "description",
    "samples",
    "seed",
    "criterion",
    "variables",
    "correlations",
    "options",
}
CLIP_KEYS = ("clip_low", "clip_high")


class ConfigError(ValueError):
    """An invalid decision file, with the location of the problem in the message."""


def _parameters(factory: Callable[..., Distribution]) -> tuple[set[str], set[str]]:
    """(all accepted, required) keyword names of a distribution constructor."""
    params = inspect.signature(factory).parameters.values()
    names = {p.name for p in params}
    return names, {p.name for p in params if p.default is inspect.Parameter.empty}


def make_distribution(name: str, spec: Any) -> Distribution:
    """Distribution for variable ``name`` from ``{dist: ..., **params}`` or a number."""
    where = f"variables.{name}"
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Fixed(spec)
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: expected a number or a mapping with 'dist', got {spec!r}")
    params = dict(spec)
    kind = params.pop("dist", None)
    clip = {key: params.pop(key) for key in CLIP_KEYS if key in params}
    if not isinstance(kind, str) or kind.lower() not in DISTRIBUTIONS:
        raise ConfigError(
            f"{where}: 'dist' must be one of {', '.join(DISTRIBUTIONS)}, got {kind!r}"
        )
    factory = DISTRIBUTIONS[kind.lower()]
    accepted, required = _parameters(factory)
    unknown, missing = set(params) - accepted, required - set(params)
    if unknown or missing:
        problems = []
        if missing:
            problems.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            problems.append(f"unknown {', '.join(sorted(unknown))}")
        raise ConfigError(
            f"{where}: {kind} takes {', '.join(sorted(accepted))} ({'; '.join(problems)})"
        )
    try:
        distribution = factory(**params)
        if clip:
            distribution = distribution.clip(clip.get("clip_low"), clip.get("clip_high"))
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where}: {error}") from None
    return distribution


def _add_correlations(decision: Decision, entries: Any) -> None:
    """``correlations: [[a, b, rho], ...]`` (Spearman rank correlations)."""
    if not isinstance(entries, list):
        raise ConfigError("'correlations' must be a list of [variable, variable, rho]")
    for i, entry in enumerate(entries):
        where = f"correlations[{i}]"
        if not isinstance(entry, list) or len(entry) != 3:
            raise ConfigError(f"{where}: expected [variable, variable, rho], got {entry!r}")
        try:
            decision.correlate(str(entry[0]), str(entry[1]), entry[2])
        except ValueError as error:
            raise ConfigError(f"{where}: {error}") from None


def from_dict(data: Mapping[str, Any]) -> Decision:
    """Build a decision from a parsed config (see ``examples/coffee_shop.yaml``)."""
    if not isinstance(data, Mapping):
        raise ConfigError("the decision file must be a mapping at the top level")
    unknown = set(data) - TOP_LEVEL
    if unknown:
        raise ConfigError(
            f"unknown top-level keys: {', '.join(sorted(map(str, unknown)))} "
            f"(allowed: {', '.join(sorted(TOP_LEVEL))})"
        )
    variables, options = data.get("variables") or {}, data.get("options")
    if not isinstance(variables, Mapping):
        raise ConfigError("'variables' must be a mapping of name -> distribution")
    if not isinstance(options, Mapping) or not options:
        raise ConfigError("'options' must be a non-empty mapping of name -> formula")
    try:
        decision = Decision(
            str(data.get("name", "Decision")),
            n=data.get("samples", 100_000),
            seed=data.get("seed"),
            criterion=data.get("criterion", "expected_value"),
        )
        for name, spec in variables.items():
            decision.uncertain(str(name), make_distribution(str(name), spec))
        if data.get("correlations") is not None:
            _add_correlations(decision, data["correlations"])
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from None
    for name, formula in options.items():
        if isinstance(formula, bool) or not isinstance(formula, (str, int, float)):
            raise ConfigError(f"options.{name}: expected a formula string or a number")
        try:
            decision.option(str(name), str(formula))
        except (FormulaError, ValueError) as error:
            raise ConfigError(f"options.{name}: {error}") from None
    return decision


def load(path: str | Path) -> Decision:
    """Read a YAML decision file."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from None
    return from_dict(data)


__all__ = ["DISTRIBUTIONS", "ConfigError", "from_dict", "load", "make_distribution"]
