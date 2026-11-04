# Correlated inputs

Real inputs rarely move independently. Demand and price, the cost of two suppliers that buy the same raw material, or a recession that hits sales and margin at once all move together. Ignoring this usually **understates risk**.

```python
d.correlate("monthly_sales", "margin", 0.5)
d.correlate(cn_cost, tr_cost, 0.4)   # Variable handles work too
```

```yaml
correlations:
  - [monthly_sales, margin, 0.5]
```

## What the number means

`rho` is a **Spearman rank correlation** between −1 and 1. A value of 0.5 means "when sales are high, margin tends to be high too". Rank correlation works for any mix of distributions, including skewed and discrete ones.

## How it works

montecarlo-hermes uses the **Iman-Conover** method:

1. every variable is sampled from its own stream, exactly as without correlation;
2. a set of correlated normal scores is generated with the requested structure;
3. each variable's samples are **reordered** to follow the ranks of its scores.

Each input therefore keeps exactly the same values and distribution. Only their pairing across variables changes, and the achieved rank correlation matches the target closely (to within ±0.01 at 50,000 samples).

## Effect on the coffee shop

| | Independent | sales ↔ margin at 0.5 |
|---|---|---|
| Mean of `open` | −6,634 | −6,096 |
| Std dev of `open` | 11,041 | 13,264 |
| P(loss) of `open` | 73.4% | 68.2% |

The correlation widens the distribution: good months are better and bad months are worse.

## Errors

Correlations must be mutually consistent. For example, "a and b are 0.9, a and c are 0.9, but b and c are −0.9" is impossible, and `run()` raises a `CorrelationError` that suggests weakening some of them. A `Fixed` variable cannot be correlated.
