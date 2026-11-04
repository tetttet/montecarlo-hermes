"""Regenerate the README images in docs/assets by running the examples.

Run:  python scripts/generate_assets.py
Checks that every YAML example and its Python twin give identical results.
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

import mchermes as mc

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
EXAMPLES = ("coffee_shop", "job_offer", "import_business")


def python_twin(name: str) -> mc.Decision:
    """Import examples/<name>.py and call its build()."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decision: mc.Decision = module.build()
    return decision


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLES:
        result = mc.load(ROOT / "examples" / f"{name}.yaml").run()
        twin = python_twin(name).run()
        for option in result.options:
            assert np.allclose(result.outcomes[option], twin.outcomes[option]), (name, option)
        driver = next(o for o in result.options if result.stats[o].std > 0)
        for theme in ("light", "dark"):
            result.plot(str(ASSETS / f"{name}_{theme}.png"), theme=theme)
            result.plot_tornado(driver, str(ASSETS / f"{name}_tornado_{theme}.png"), theme=theme)
            result.plot_cdf(str(ASSETS / f"{name}_cdf_{theme}.png"), theme=theme)
        print(f"{name}: best = {result.best_option()}  (YAML and Python agree)")


if __name__ == "__main__":
    main()
