# Distributions

Every distribution validates its parameters and exposes analytic `mean` and `std`. Use `.draw(n, seed)` to look at a few samples:

```python
mc.PERT(1, 2, 6).draw(5, seed=0)
```

## Continuous

| Distribution | Python | YAML | Typical use |
|---|---|---|---|
| Normal | `Normal(mean, std)` | `{dist: normal, mean, std}` | Symmetric noise around a best guess |
| LogNormal | `LogNormal(mean, std)` | `{dist: lognormal, mean, std}` | Positive, right-skewed amounts |
| Uniform | `Uniform(low, high)` | `{dist: uniform, low, high}` | "Somewhere in between" |
| Triangular | `Triangular(low, mode, high)` | `{dist: triangular, low, mode, high}` | Quick three-point estimates |
| PERT | `PERT(low, mode, high, lamb=4)` | `{dist: pert, low, mode, high}` | Smoother three-point estimates |
| Beta | `Beta(alpha, beta, low=0, high=1)` | `{dist: beta, alpha, beta}` | Rates and shares |
| Exponential | `Exponential(mean)` | `{dist: exponential, mean}` | Waiting times |
| Gamma | `Gamma(shape, scale)` | `{dist: gamma, shape, scale}` | Positive skewed amounts, claim sizes |
| Weibull | `Weibull(shape, scale)` | `{dist: weibull, shape, scale}` | Lifetimes, time to failure |
| Student t | `StudentT(df, loc=0, scale=1)` | `{dist: student_t, df}` | Fat-tailed returns and shocks |

`LogNormal` takes the mean and std of the variable itself. `LogNormal.from_log_params(mu, sigma)` and `Gamma.from_mean_std(mean, std)` cover the other common parameterizations.

## Discrete

| Distribution | Python | YAML | Typical use |
|---|---|---|---|
| Bernoulli | `Bernoulli(p)` | `{dist: bernoulli, p}` | "Event happens with probability p" (1 or 0) |
| Binomial | `Binomial(trials, p)` | `{dist: binomial, trials, p}` | Successes out of n attempts |
| Poisson | `Poisson(lam)` | `{dist: poisson, lam}` | Number of events in a period |
| Discrete | `Discrete(values, probabilities)` | `{dist: discrete, values, probabilities}` | A handful of known scenarios |

## From data and constants

| Distribution | Python | YAML | Typical use |
|---|---|---|---|
| Empirical | `Empirical(data)` | `{dist: empirical, data: [...]}` | Bootstrap from history |
| Fixed | `Fixed(value)` | `value` or `{dist: fixed, value}` | Known constants |

## Bounding a distribution

`.clip(low, high)` sets values outside the bounds to the bound. Use it, for example, to stop a normal price going negative:

```python
price = d.uncertain("price", mc.Normal(10, 4).clip(low=0))
```

In YAML: `{dist: normal, mean: 10, std: 4, clip_low: 0}`.

!!! tip "Bernoulli events"
    Multiply an outcome by a Bernoulli to switch it on and off: `delay * delay_cost` costs `delay_cost` in the futures where the delay happens, and nothing otherwise.
