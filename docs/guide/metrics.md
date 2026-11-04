# Metrics and criteria

`r.stats[option]` is an `OptionStats` with:

| Metric | Attribute | Meaning |
|---|---|---|
| Mean | `mean` | Expected outcome |
| MC error | `mean_se` | Standard error of the mean from sampling |
| Median, std | `median`, `std` | Center and spread |
| Min, max | `min`, `max` | Most extreme simulated futures |
| Percentiles | `p5`, `p10`, `p50`, `p90`, `p95` | E.g. P10: 90% of futures do better |
| P(loss) | `prob_loss` | P(outcome < 0) |
| VaR 95% | `var_95` | 5th percentile: you do at least this well with 95% confidence |
| CVaR 95% | `cvar_95` | Average of the worst 5% of futures |
| P(best) | `prob_best` | Share of futures in which this option beats all others (ties split) |
| Expected regret | `expected_regret` | Average shortfall against the best option in each future |

All values are in outcome units, so a negative VaR is a loss.

More questions you can ask:

```python
r.prob_loss("open", threshold=-10_000)  # P(lose more than 10k)
r.prob_above("open", 10_000)            # P(make more than 10k)
r.percentile("open", [1, 99])
r.compare("open", "dont_open")          # distribution of the difference
```

`compare(a, b)` returns `prob_a_better`, `mean_difference` and P5/P50/P95 of `a − b`, computed sample by sample on the same futures.

## Decision criteria

| Criterion | Picks the option that... | Good when |
|---|---|---|
| `expected_value` (default) | has the highest mean | you make many similar bets |
| `risk_averse` | has the highest P10 | a bad outcome is very costly |
| `min_regret` | has the lowest expected regret | you want to avoid hindsight pain |

```python
r.best_option()                 # decision's default criterion
r.best_option("risk_averse")
mc.Decision("...", criterion="min_regret")
```

Criteria can disagree. In `examples/job_offer.yaml`, `accept` wins on expected value and regret, while `stay` wins under `risk_averse` because of the layoff risk.
