"""The safe formula parser: what it computes and what it refuses."""

import numpy as np
import pytest

from mchermes.formula import FormulaError, compile_formula

ENV = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 2.0, 1.0])}


def run(source: str) -> np.ndarray:
    return np.asarray(compile_formula(source, ENV).evaluate(ENV))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("a + b * 2 - 1", [6, 5, 4]),
        ("a / b", [1 / 3, 1, 3]),
        ("a // 2 + a % 2", [1, 1, 2]),
        ("a ** 2", [1, 4, 9]),
        ("-a + +b", [2, 0, -2]),
        ("a > b", [0, 0, 1]),
        ("a >= 2", [0, 1, 1]),
        ("a < 2", [1, 0, 0]),
        ("a <= 2", [1, 1, 0]),
        ("a == 2", [0, 1, 0]),
        ("a != 2", [1, 0, 1]),
        ("1 < a < 3", [0, 1, 0]),
        ("a > 1 and b > 1", [0, 1, 0]),
        ("a > 2 or b > 2", [1, 0, 1]),
        ("not a > 1", [1, 0, 0]),
        ("where(a > b, a, b)", [3, 2, 3]),
        ("min(a, b)", [1, 2, 1]),
        ("max(a, b, 2.5)", [3, 2.5, 3]),
        ("abs(a - b)", [2, 0, 2]),
        ("log(exp(a))", [1, 2, 3]),
        ("sqrt(a * a)", [1, 2, 3]),
        ("True + 1", [2, 2, 2]),
        ("(a > b) * 10", [0, 0, 10]),
    ],
)
def test_evaluates(source: str, expected: list[float]) -> None:
    np.testing.assert_allclose(np.broadcast_to(run(source), (3,)), expected)


def test_tracks_used_names() -> None:
    formula = compile_formula("a * 2", ENV)
    assert formula.names == frozenset({"a"})
    assert formula.source == "a * 2"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("__import__('os')", "'__import__' is not allowed"),
        ("__import__('os').system('echo hi')", "only direct calls"),
        ("__builtins__", "unknown variable"),
        ("a.__class__", "Attribute is not allowed"),
        ("a.real", "Attribute is not allowed"),
        ("open('x')", "'open' is not allowed"),
        ("eval('1')", "'eval' is not allowed"),
        ("np.exp(a)", "only direct calls"),
        ("a[0]", "Subscript is not allowed"),
        ("'text'", "only numbers are allowed"),
        ("lambda: 1", "Lambda is not allowed"),
        ("[a, b]", "List is not allowed"),
        ("a if b else 1", "IfExp is not allowed"),
        ("(x := 1)", "NamedExpr is not allowed"),
        ("a @ b", "MatMult is not allowed"),
        ("~a", "Invert is not allowed"),
        ("a in b", "In is not allowed"),
        ("a is b", "Is is not allowed"),
        ("max(a=1, b=2)", "keyword arguments"),
        ("max(*a)", r"\*arguments"),
        ("sqrt(a, b)", "takes 1 arguments"),
        ("max(a)", "takes 2 or more arguments"),
        ("where(a, b)", "takes 3 arguments"),
        ("exp", "is a function"),
        ("c + 1", "unknown variable 'c'"),
        ("aa + 1", "did you mean 'a'"),
        ("1e999", "finite"),
        ("a +", "invalid syntax"),
        ("", "must not be empty"),
        ("   ", "must not be empty"),
        ("1" * 10_001, "at most"),
        ("-" * 5000 + "1", "nested too deeply"),
    ],
)
def test_rejects(source: str, message: str) -> None:
    with pytest.raises(FormulaError, match=message):
        compile_formula(source, ENV)


def test_rejects_non_string() -> None:
    with pytest.raises(FormulaError, match="must be a string"):
        compile_formula(42, ENV)  # type: ignore[arg-type]


def test_formula_error_is_value_error() -> None:
    assert issubclass(FormulaError, ValueError)


def test_error_reports_column() -> None:
    with pytest.raises(FormulaError, match="column 5"):
        compile_formula("a + zz", ENV)
