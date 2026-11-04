"""montecarlo-hermes: simulate your decisions before you make them.

>>> import mchermes as mc
>>> d = mc.Decision("Raise prices?", n=10_000, seed=1)
>>> demand = d.uncertain("demand", mc.Normal(1000, 100))
>>> _ = d.option("raise", lambda: demand * 0.9 * 12).option("keep", lambda: demand * 10)
>>> d.run().best_option()
'raise'
"""

from mchermes.analysis import Comparison, InformationValue
from mchermes.correlation import CorrelationError
from mchermes.decision import Decision, Variable
from mchermes.distributions import (
    PERT,
    Bernoulli,
    Beta,
    Binomial,
    Clipped,
    Discrete,
    Distribution,
    Empirical,
    Exponential,
    Fixed,
    Gamma,
    LogNormal,
    Normal,
    Poisson,
    StudentT,
    Triangular,
    Uniform,
    Weibull,
)
from mchermes.formula import FormulaError
from mchermes.loader import ConfigError, from_dict, load
from mchermes.metrics import OptionStats
from mchermes.result import Result
from mchermes.sensitivity import Sensitivity
from mchermes.sweep import Sweep

__version__ = "0.2.0"

__all__ = [
    "PERT",
    "Bernoulli",
    "Beta",
    "Binomial",
    "Clipped",
    "Comparison",
    "ConfigError",
    "CorrelationError",
    "Decision",
    "Discrete",
    "Distribution",
    "Empirical",
    "Exponential",
    "Fixed",
    "FormulaError",
    "Gamma",
    "InformationValue",
    "LogNormal",
    "Normal",
    "OptionStats",
    "Poisson",
    "Result",
    "Sensitivity",
    "StudentT",
    "Sweep",
    "Triangular",
    "Uniform",
    "Variable",
    "Weibull",
    "__version__",
    "from_dict",
    "load",
]
