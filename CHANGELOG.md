# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-04

### Added
- `Decision` API: declare uncertain inputs with `uncertain()`, options with `option()`
  (lambda, safe formula string or number), and `run()` everything vectorized with NumPy.
- Distributions: Normal, LogNormal, Uniform, Triangular, PERT, Beta, Bernoulli,
  Discrete and Fixed, each validating its parameters.
- Reproducible sampling: every variable owns a random stream derived from
  `(seed, name)`.
- Metrics per option: mean, median, std, min/max, P5/P10/P50/P90/P95, P(loss),
  VaR and CVaR at 95%, probability of being best and expected regret.
- Decision criteria: `expected_value`, `risk_averse` (max P10) and `min_regret`.
- Tornado sensitivity from Spearman rank correlations.
- YAML decision files with a safe `ast`-based formula parser (no `eval`).
- `mchermes` CLI: `run` (table, `--json`, `--plot`) and `init`.
- Charts in light and dark themes: distribution overlay, tornado and S-curve.
- Examples: coffee shop, job offer and import business, each in YAML and Python.

[Unreleased]: https://github.com/tetttet/montecarlo-hermes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tetttet/montecarlo-hermes/releases/tag/v0.1.0
