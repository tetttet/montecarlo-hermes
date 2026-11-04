"""Open a second coffee shop? The Python twin of coffee_shop.yaml.

Run:  python examples/coffee_shop.py
"""

import mchermes as mc


def build() -> mc.Decision:
    """The decision: yearly profit of opening versus doing nothing."""
    d = mc.Decision("Open a second coffee shop?", n=100_000, seed=42)
    rent = d.uncertain("rent", mc.Normal(3000, 300))
    sales = d.uncertain("monthly_sales", mc.Triangular(8000, 12000, 18000))
    margin = d.uncertain("margin", mc.Uniform(0.25, 0.4))

    d.option("open", lambda: sales * margin * 12 - rent * 12 - 20_000)
    d.option("dont_open", lambda: 0)
    return d


if __name__ == "__main__":
    result = build().run()
    result.summary()
    print(f"\nP(loss) if we open: {result.prob_loss('open'):.1%}")
    for s in result.sensitivity("open"):
        print(f"  {s.variable:<14} {s.spearman:+.2f}")
    result.plot("coffee_shop.png")
    result.plot_tornado("open", "coffee_shop_tornado.png")
    print("\nCharts saved: coffee_shop.png, coffee_shop_tornado.png")
