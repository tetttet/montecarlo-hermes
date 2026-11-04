# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-04

### Added
- Correlated inputs: `Decision.correlate(a, b, rho)` and a YAML `correlations:` list, using
  Spearman targets via Iman-Conover rank reordering (marginal distributions untouched).
- Distributions: Exponential, Gamma (with `from_mean_std`), Weibull, StudentT, Poisson,
  Binomial and Empirical, plus `Distribution.clip()` and `Distribution.draw()`.
- Value of information: `Result.evpi()` and `Result.value_of_information()` (EVPPI per input).
- What-if analysis: `Decision.sweep()` with `Sweep.breakeven()` and a sweep chart;
  `Result.given()` for conditional results; `Result.compare()` and `Result.prob_above()`.
- Standalone HTML report (`Result.to_html()`, `mchermes run --html`), Jupyter display and
  `Result.to_csv()`.
- CLI: `mchermes sweep`, `mchermes validate`, and `--html`, `--csv` and `--given` for `run`.
- `OptionStats.mean_se`, the Monte Carlo standard error of the mean.
- `Decision.from_yaml()` and `Decision.from_dict()`.
- Documentation site, quickstart notebook, benchmark script, Python 3.13 support.

### Changed
- Much faster analysis at large sample sizes: cached ranks, capped rank statistics and
  partition-based CVaR. At 1M samples `to_dict()` went from 3.3 s to 0.18 s.
- The terminal summary also reports the value of perfect information.
- `examples/import_business` now correlates the two suppliers' unit costs.

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

[Unreleased]: https://github.com/tetttet/montecarlo-hermes/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tetttet/montecarlo-hermes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tetttet/montecarlo-hermes/releases/tag/v0.1.0
