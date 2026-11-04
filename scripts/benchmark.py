"""Time the main operations on a 10-input, 3-option decision.

Run:  python scripts/benchmark.py [samples]
"""

import sys
import time
from collections.abc import Callable

import mchermes as mc


def timed(label: str, function: Callable[[], object]) -> object:
    start = time.perf_counter()
    value = function()
    print(f"  {label:<26}{(time.perf_counter() - start) * 1000:9.1f} ms")
    return value


def main(n: int) -> None:
    d = mc.Decision("Benchmark", n=n, seed=1)
    kinds = [mc.Normal(10, 2), mc.Triangular(1, 2, 5), mc.Uniform(0, 1), mc.Bernoulli(0.3)]
    xs = [d.uncertain(f"x{i}", kinds[i % len(kinds)]) for i in range(10)]
    d.option("a", lambda: sum((i + 1) * x for i, x in enumerate(xs)))
    d.option("b", "x0 * x1 - x2 * 100 + where(x3 > 0, 5, 0)")
    d.option("c", 50)
    d.correlate("x0", "x1", 0.5)
    print(f"{n:,} samples, 10 inputs, 3 options, 1 correlation")
    result = timed("run (sample + evaluate)", d.run)
    assert isinstance(result, mc.Result)
    timed("metrics", lambda: result.stats)
    timed("sensitivity (one option)", lambda: result.sensitivity("a"))
    timed("value of information", result.value_of_information)
    timed("to_dict (everything)", result.to_dict)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000)
