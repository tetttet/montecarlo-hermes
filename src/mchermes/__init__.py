"""montecarlo-hermes: simulate your decisions before you make them.

>>> import mchermes as mc
>>> d = mc.Decision("Raise prices?", n=10_000, seed=1)
>>> demand = d.uncertain("demand", mc.Normal(1000, 100))
>>> _ = d.option("raise", lambda: demand * 0.9 * 12).option("keep", lambda: demand * 10)
>>> d.run().best_option()
'raise'
"""

from mchermes.decision import Decision, Variable
from mchermes.distributions import (
    PERT,
    Bernoulli,
    Beta,
    Discrete,
    Distribution,
    Fixed,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
)
from mchermes.formula import FormulaError
from mchermes.metrics import OptionStats
from mchermes.result import Result
from mchermes.sensitivity import Sensitivity

__version__ = "0.1.0"

__all__ = [
    "PERT",
    "Bernoulli",
    "Beta",
    "Decision",
    "Discrete",
    "Distribution",
    "Fixed",
    "FormulaError",
    "LogNormal",
    "Normal",
    "OptionStats",
    "Result",
    "Sensitivity",
    "Triangular",
    "Uniform",
    "Variable",
    "__version__",
]
