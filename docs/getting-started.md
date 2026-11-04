# Getting started

## Install

```bash
pip install montecarlo-hermes            # core
pip install "montecarlo-hermes[pandas]"  # adds to_dataframe()
```

montecarlo-hermes needs Python 3.10 or newer. The import name is `mchermes`.

## Your first decision

A decision has three parts:

1. **Uncertain inputs**: each one gets a distribution.
2. **Options**: each one gets an outcome formula over the inputs.
3. **A run**: the library samples every input, evaluates every option on all samples at once, and summarizes the result.

```python
import mchermes as mc

d = mc.Decision("Open a second coffee shop?", n=100_000, seed=42)
rent   = d.uncertain("rent", mc.Normal(3000, 300))
sales  = d.uncertain("monthly_sales", mc.Triangular(8000, 12000, 18000))
margin = d.uncertain("margin", mc.Uniform(0.25, 0.4))

d.option("open",      lambda: sales * margin * 12 - rent * 12 - 20_000)
d.option("dont_open", lambda: 0)

r = d.run()
r.summary()             # table of metrics + recommendation
r.best_option()         # 'dont_open'
r.prob_loss("open")     # 0.7336
```

`rent`, `sales` and `margin` behave like NumPy arrays that hold every sample, so a formula runs once over all 100,000 futures, with no Python loops.

## Same decision without code

Save this as `coffee_shop.yaml`:

```yaml
name: Open a second coffee shop?
samples: 100000
seed: 42
variables:
  rent: {dist: normal, mean: 3000, std: 300}
  monthly_sales: {dist: triangular, low: 8000, mode: 12000, high: 18000}
  margin: {dist: uniform, low: 0.25, high: 0.4}
options:
  open: "monthly_sales * margin * 12 - rent * 12 - 20000"
  dont_open: "0"
```

```bash
mchermes run coffee_shop.yaml
mchermes run coffee_shop.yaml --html report.html
```

## Next steps

- Choose distributions: [Distributions](guide/distributions.md)
- Read the numbers: [Metrics and criteria](guide/metrics.md)
- Find what matters: [Sensitivity and value of information](guide/sensitivity.md)
- Explore scenarios: [What-if analysis](guide/what-if.md)
