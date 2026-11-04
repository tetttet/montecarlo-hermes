"""Stay at the current job or take the new offer? The Python twin of job_offer.yaml.

Run:  python examples/job_offer.py
"""

import mchermes as mc


def build() -> mc.Decision:
    """The decision: total compensation over the next two years."""
    d = mc.Decision("Accept the new job offer?", n=100_000, seed=7)
    salary = d.uncertain("current_salary", mc.Normal(92_000, 2_500))
    raise_pct = d.uncertain("raise_pct", mc.Uniform(0.02, 0.05))
    new_base = d.uncertain("new_base", mc.Fixed(96_000))
    bonus_paid = d.uncertain("bonus_paid", mc.Bernoulli(0.5))
    bonus = d.uncertain("bonus", mc.PERT(5_000, 12_000, 30_000))
    equity = d.uncertain("equity", mc.LogNormal(mean=8_000, std=12_000))
    relocation = d.uncertain("relocation", mc.Triangular(3_000, 6_000, 12_000))
    layoff = d.uncertain("layoff", mc.Bernoulli(0.15))
    months = d.uncertain("months_unemployed", mc.Discrete([2, 4, 6], [0.5, 0.3, 0.2]))

    d.option("stay", lambda: salary * (1 + raise_pct) + salary * (1 + raise_pct) ** 2)
    d.option(
        "accept",
        lambda: (
            2 * new_base
            + 2 * bonus_paid * bonus
            + equity
            - relocation
            - layoff * months / 12 * new_base
        ),
    )
    return d


if __name__ == "__main__":
    result = build().run()
    result.summary()
    for criterion in ("expected_value", "risk_averse", "min_regret"):
        print(f"{criterion:>15}: {result.best_option(criterion)}")
    result.plot("job_offer.png")
    result.plot_cdf("job_offer_cdf.png")
    print("\nCharts saved: job_offer.png, job_offer_cdf.png")
