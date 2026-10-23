# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A bounded arithmetic evaluator for model-provided calculator arguments."""

import ast
import math
import operator
from typing import cast

MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 64
MAX_OPERATIONS = 16
MAX_DEPTH = 12
MAX_EXPONENT = 12
MAX_ABS_VALUE = 1_000_000_000_000

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _bounded_number(value: object) -> int | float:
    # bool is an int subclass, but is not an arithmetic literal for this tool.
    if type(value) not in (int, float):
        raise ValueError("Only real numeric literals are allowed")
    number = cast(int | float, value)
    if abs(number) > MAX_ABS_VALUE or (
        isinstance(number, float) and not math.isfinite(number)
    ):
        raise ValueError("Numeric value exceeds the calculator limit")
    return number


def evaluate_arithmetic(expression: str) -> int | float:
    """Evaluate numbers, parentheses, +, -, *, /, //, %, ** and unary +/-.

    Names, attributes, calls, containers and all other Python syntax are rejected.
    Limits apply to input, syntax size/depth, operation count, exponent magnitude
    and every intermediate result. No Python code is compiled or executed.
    """
    if type(expression) is not str:
        raise ValueError("Expression must be a string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression exceeds the calculator length limit")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError):
        raise ValueError("Invalid arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression exceeds the calculator syntax limit")

    operations = 0

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        nonlocal operations
        if depth > MAX_DEPTH:
            raise ValueError("Expression exceeds the calculator depth limit")
        if isinstance(node, ast.Constant):
            return _bounded_number(node.value)
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            operations += 1
            if operations > MAX_OPERATIONS:
                raise ValueError("Expression exceeds the calculator operation limit")
            try:
                if isinstance(node, ast.UnaryOp):
                    unary = _UNARY_OPERATORS.get(type(node.op))
                    if unary is not None:
                        return _bounded_number(unary(evaluate(node.operand, depth + 1)))
                else:
                    left = evaluate(node.left, depth + 1)
                    right = evaluate(node.right, depth + 1)
                    if isinstance(node.op, ast.Pow):
                        if abs(right) > MAX_EXPONENT:
                            raise ValueError("Exponent exceeds the calculator limit")
                        # Both operands and the exponent are bounded before pow.
                        return _bounded_number(left**right)
                    binary = _BINARY_OPERATORS.get(type(node.op))
                    if binary is not None:
                        return _bounded_number(binary(left, right))
            except ArithmeticError:
                raise ValueError(
                    "Arithmetic operation is undefined or too large"
                ) from None
        raise ValueError("Only arithmetic operators and numeric literals are allowed")

    return evaluate(tree.body)
