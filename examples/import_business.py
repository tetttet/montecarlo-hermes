"""Import from China or Turkey? The Python twin of import_business.yaml.

Run:  python examples/import_business.py
"""

import mchermes as mc


def build() -> mc.Decision:
    """The decision: profit on one container of goods."""
    d = mc.Decision("Import from China or Turkey?", n=100_000, seed=2024)
    units = d.uncertain("units", mc.Fixed(10_000))
    price = d.uncertain("sale_price", mc.Normal(9.5, 0.8))

    cn_cost = d.uncertain("china_unit_cost", mc.Normal(4.2, 0.3))
    cn_ship = d.uncertain("china_shipping", mc.Triangular(3_500, 5_000, 9_000))
    cn_duty = d.uncertain("china_duty", mc.Uniform(0.15, 0.25))
    cn_delay = d.uncertain("china_delay", mc.Bernoulli(0.25))

    tr_cost = d.uncertain("turkey_unit_cost", mc.Normal(5.1, 0.25))
    tr_ship = d.uncertain("turkey_shipping", mc.Triangular(1_200, 1_800, 3_000))
    tr_duty = d.uncertain("turkey_duty", mc.Uniform(0.02, 0.06))
    tr_delay = d.uncertain("turkey_delay", mc.Bernoulli(0.08))

    delay_cost = d.uncertain("delay_cost", mc.Triangular(2_000, 6_000, 15_000))

    # Both suppliers buy the same raw materials: their prices move together.
    d.correlate(cn_cost, tr_cost, 0.4)

    revenue = lambda: units * price  # noqa: E731 - shared by both options
    d.option(
        "china",
        lambda: revenue() - units * cn_cost * (1 + cn_duty) - cn_ship - cn_delay * delay_cost,
    )
    d.option(
        "turkey",
        lambda: revenue() - units * tr_cost * (1 + tr_duty) - tr_ship - tr_delay * delay_cost,
    )
    return d


if __name__ == "__main__":
    result = build().run()
    result.summary()
    print(f"\nP(China beats Turkey): {result.stats['china'].prob_best:.1%}")
    result.plot("import_business.png")
    result.plot_tornado("china", "import_business_tornado.png")
    print("\nCharts saved: import_business.png, import_business_tornado.png")
