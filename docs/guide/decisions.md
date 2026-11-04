# Decisions and options

## Declaring inputs

```python
d = mc.Decision("Open a second coffee shop?", n=100_000, seed=42)
sales = d.uncertain("monthly_sales", mc.Triangular(8000, 12000, 18000))
```

`uncertain()` returns a `Variable`. It behaves like a NumPy array of samples: arithmetic, comparisons and NumPy functions (`np.maximum`, `np.where`, `np.exp`, ...) work on every sample at once. Names must be valid identifiers and unique within a decision.

## Defining options

An option's outcome can be:

| Form | Example | Notes |
|---|---|---|
| Zero-argument callable | `lambda: sales * margin * 12 - rent * 12` | Any vectorized NumPy code |
| Safe formula string | `"monthly_sales * margin * 12 - rent * 12"` | Same language as YAML files |
| Number | `0` | A certain outcome |

```python
d.option("open", lambda: sales * margin * 12 - rent * 12 - 20_000)
d.option("lease", "monthly_sales * margin * 12 - 30000")
d.option("dont_open", 0)
```

Options are evaluated once per run on whole arrays. An outcome must be a number or contain one value per sample, and it must be finite. A NaN from `log` of a negative number raises a clear error instead of corrupting the metrics.

## Running

```python
r = d.run()                  # uses n and seed from the Decision
r = d.run(n=1_000_000)       # more samples
r = d.run(seed=7)            # another reproducible run
```

### Reproducibility

Each variable draws from its own random stream, derived from `(seed, variable name)`. As a result:

- the same seed gives identical results;
- adding, removing or reordering variables never changes the draws of the others;
- options are compared on **common random numbers**: every option sees the same futures.

With `seed=None`, a fresh seed is drawn and recorded on `r.seed`, so any run can be replayed.

### How many samples?

`r.stats[option].mean_se` is the Monte Carlo standard error of the mean. For the coffee shop example with 100,000 samples it is about 35, against a mean of −6,634. Ten times more samples shrink it by about √10 ≈ 3.2×.
