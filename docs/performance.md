# Performance

Everything is vectorized NumPy: inputs are sampled as arrays, formulas are evaluated once over all samples, and metrics use partial sorts.

Measured with `python scripts/benchmark.py` on an Apple M3 Pro, with a decision of 10 inputs, 3 options and 1 correlation:

| Operation | 100,000 samples | 1,000,000 samples |
|---|---:|---:|
| `run()`: sample and evaluate | 39 ms | 320 ms |
| Metrics for every option | 9 ms | 60 ms |
| Sensitivity for one option | 63 ms | 139 ms |
| Value of information (all inputs) | 65 ms | 142 ms |
| `to_dict()`: everything | 79 ms | 178 ms |

## Design choices that keep it fast

- **One stream per variable.** Sampling is a single NumPy call per input.
- **Capped rank statistics.** Spearman correlations and EVPPI use at most 200,000 samples. Their standard error is then below 0.003, and their cost stays flat as you add samples.
- **Cached ranks.** Input ranks are computed once per result and reused for every option.
- **Partial sorts.** CVaR uses `np.partition` instead of a full sort.
- **Correlation by reordering.** Iman-Conover needs one sort per correlated input, with no inverse CDFs.

## Tips

- Start with `n=10_000` while building a model, then raise it to 100k–1M for the final numbers. `stats[option].mean_se` tells you whether you have enough.
- Keep option formulas vectorized: use `np.where` and `np.maximum` rather than Python `if` and `max`.
