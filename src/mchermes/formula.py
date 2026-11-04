"""Safe outcome formulas: text compiled to vectorized NumPy closures.

Formulas are parsed with :mod:`ast` and only a small whitelist of nodes is compiled:
numbers, variable names, arithmetic (``+ - * / // % **``), comparisons, ``and`` /
``or`` / ``not`` and calls of :data:`FUNCTIONS`. Attributes, subscripts, strings,
keyword arguments and every other construct are rejected. Nothing is ever passed to
``eval`` or ``exec``.
"""

from __future__ import annotations

import ast
import difflib
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import reduce

import numpy as np
from numpy.typing import NDArray

Value = NDArray[np.float64] | float
Env = Mapping[str, NDArray[np.float64]]
Node = Callable[[Env], Value]

MAX_LENGTH = 10_000


class FormulaError(ValueError):
    """A formula that cannot be parsed or uses something outside the whitelist."""


def _where(condition: Value, a: Value, b: Value) -> Value:
    return np.where(np.asarray(condition) != 0, a, b)


# name -> (minimum arguments, maximum arguments, implementation)
FUNCTIONS: dict[str, tuple[int, int, Callable[..., Value]]] = {
    "min": (2, 32, lambda *a: reduce(np.minimum, a)),
    "max": (2, 32, lambda *a: reduce(np.maximum, a)),
    "where": (3, 3, _where),
    "abs": (1, 1, np.abs),
    "log": (1, 1, np.log),
    "exp": (1, 1, np.exp),
    "sqrt": (1, 1, np.sqrt),
}

_BINARY: dict[type[ast.operator], Callable[[Value, Value], Value]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_COMPARE: dict[type[ast.cmpop], Callable[[Value, Value], Value]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _as_float(value: object) -> Value:
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class Formula:
    """A compiled formula; call :meth:`evaluate` with arrays for every name it uses."""

    source: str
    names: frozenset[str]
    _root: Node

    def evaluate(self, env: Env) -> Value:
        """Evaluate on all samples at once; returns a float64 array or scalar."""
        with np.errstate(all="ignore"):
            return _as_float(self._root(env))


class _Compiler:
    def __init__(self, source: str, allowed: frozenset[str]) -> None:
        self.source, self.allowed, self.used = source, allowed, set[str]()

    def fail(self, message: str, node: ast.AST | None = None) -> FormulaError:
        where = f" (column {node.col_offset + 1})" if isinstance(node, ast.expr) else ""
        return FormulaError(f"{message}{where} in formula {self.source!r}")

    def visit(self, node: ast.AST) -> Node:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise self.fail(f"{type(node).__name__} is not allowed", node)
        result: Node = method(node)
        return result

    def visit_Expression(self, node: ast.Expression) -> Node:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Node:
        if isinstance(node.value, bool):
            value = float(node.value)
        elif isinstance(node.value, (int, float)):
            value = float(node.value)
            if not math.isfinite(value):
                raise self.fail("numbers must be finite", node)
        else:
            raise self.fail(f"only numbers are allowed, got {node.value!r}", node)
        return lambda env: value

    def visit_Name(self, node: ast.Name) -> Node:
        name = node.id
        if name in FUNCTIONS:
            raise self.fail(f"{name!r} is a function; call it like {name}(...)", node)
        if name not in self.allowed:
            close = difflib.get_close_matches(name, sorted(self.allowed), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise self.fail(f"unknown variable {name!r}{hint}", node)
        self.used.add(name)
        return lambda env: env[name]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Node:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return lambda env: -_as_float(operand(env))
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return lambda env: _as_float(np.asarray(operand(env)) == 0)
        raise self.fail(f"operator {type(node.op).__name__} is not allowed", node)

    def visit_BinOp(self, node: ast.BinOp) -> Node:
        function = _BINARY.get(type(node.op))
        if function is None:
            raise self.fail(f"operator {type(node.op).__name__} is not allowed", node)
        left, right = self.visit(node.left), self.visit(node.right)
        return lambda env: function(_as_float(left(env)), _as_float(right(env)))

    def visit_BoolOp(self, node: ast.BoolOp) -> Node:
        parts = [self.visit(v) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or

        def run(env: Env) -> Value:
            flags = (np.asarray(p(env)) != 0 for p in parts)
            return _as_float(reduce(combine, flags))

        return run

    def visit_Compare(self, node: ast.Compare) -> Node:
        operands = [self.visit(node.left), *(self.visit(c) for c in node.comparators)]
        checks = []
        for op in node.ops:
            function = _COMPARE.get(type(op))
            if function is None:
                raise self.fail(f"comparison {type(op).__name__} is not allowed", node)
            checks.append(function)

        def run(env: Env) -> Value:
            values = [_as_float(o(env)) for o in operands]
            pairs = (f(a, b) for f, a, b in zip(checks, values, values[1:], strict=False))
            return _as_float(reduce(np.logical_and, pairs))

        return run

    def visit_Call(self, node: ast.Call) -> Node:
        if not isinstance(node.func, ast.Name):
            raise self.fail("only direct calls of whitelisted functions are allowed", node)
        name = node.func.id
        if name not in FUNCTIONS:
            allowed = ", ".join(sorted(FUNCTIONS))
            raise self.fail(f"function {name!r} is not allowed (allowed: {allowed})", node)
        if node.keywords:
            raise self.fail(f"{name}() does not take keyword arguments", node)
        low, high, function = FUNCTIONS[name]
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise self.fail(f"{name}() does not take *arguments", node)
        if not low <= len(node.args) <= high:
            expected = str(low) if low == high else f"{low} or more"
            raise self.fail(f"{name}() takes {expected} arguments, got {len(node.args)}", node)
        args = [self.visit(a) for a in node.args]
        return lambda env: function(*(_as_float(a(env)) for a in args))


def compile_formula(source: str, names: Iterable[str]) -> Formula:
    """Compile ``source`` into a :class:`Formula` that may reference only ``names``.

    Raises:
        FormulaError: on a syntax error or anything outside the whitelist.
    """
    if not isinstance(source, str):
        raise FormulaError(f"a formula must be a string, got {type(source).__name__}")
    text = source.strip()
    if not text:
        raise FormulaError("a formula must not be empty")
    if len(text) > MAX_LENGTH:
        raise FormulaError(f"a formula must be at most {MAX_LENGTH} characters")
    compiler = _Compiler(text, frozenset(names))
    try:
        root = compiler.visit(ast.parse(text, mode="eval"))
    except SyntaxError as error:
        raise FormulaError(f"invalid syntax in formula {text!r}: {error.msg}") from None
    except RecursionError:
        raise FormulaError("formula is nested too deeply") from None
    return Formula(text, frozenset(compiler.used), root)


__all__ = ["FUNCTIONS", "Formula", "FormulaError", "compile_formula"]
