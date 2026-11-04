# YAML files

A decision file describes the full model without code:

```yaml
name: Import from China or Turkey?   # shown in tables, charts and reports
samples: 100000                      # optional, default 100000
seed: 2024                           # optional; omit for a fresh random seed
criterion: expected_value            # optional: expected_value | risk_averse | min_regret

variables:
  units: 10000                                   # a plain number is fixed
  sale_price: {dist: normal, mean: 9.5, std: 0.8}
  china_unit_cost: {dist: normal, mean: 4.2, std: 0.3}
  turkey_unit_cost: {dist: normal, mean: 5.1, std: 0.25}
  china_delay: {dist: bernoulli, p: 0.25}
  delay_cost: {dist: triangular, low: 2000, mode: 6000, high: 15000}
  discount: {dist: normal, mean: 0.05, std: 0.04, clip_low: 0}

correlations:                                    # optional, Spearman rank correlations
  - [china_unit_cost, turkey_unit_cost, 0.4]

options:
  china: "units * sale_price - units * china_unit_cost - china_delay * delay_cost"
  turkey: "units * sale_price - units * turkey_unit_cost"
```

Load and run it from Python with `mc.load("file.yaml").run()` or `mc.Decision.from_yaml(...)`, or from the shell with `mchermes run file.yaml`.

## Formula language

Formulas are parsed with Python's `ast` module and compiled to vectorized NumPy. **They are never passed to `eval`.** Only the following is allowed:

| Allowed | Examples |
|---|---|
| Numbers and variable names | `12`, `0.5`, `monthly_sales` |
| Arithmetic | `+ - * / // % **` |
| Comparisons (yield 1 or 0) | `< <= > >= == !=`, chained `0 < x < 1` |
| Logic | `and`, `or`, `not` |
| Functions | `min(a, b, ...)`, `max(a, b, ...)`, `where(cond, a, b)`, `abs`, `log`, `exp`, `sqrt` |

Anything else is rejected with an error that names the problem and its column: attributes (`x.__class__`), subscripts, strings, lambdas, `__import__` and calls to any other function. Unknown names come with a suggestion (`did you mean 'margin'?`).

## Validation

`mchermes validate file.yaml` checks a file and runs 100 samples. Errors give the exact location:

```text
error: variables.rent: normal takes mean, std (missing std)
error: options.open: unknown variable 'monthy_sales' (did you mean 'monthly_sales'?) (column 1) ...
error: correlations[0]: correlation 'a'-'b' must be strictly between -1 and 1, got 1.2
```
