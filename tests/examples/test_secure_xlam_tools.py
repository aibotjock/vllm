# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with unittest; no vLLM, OpenAI, GPU or model dependencies are imported."""

import ast
import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIRECTORY = ROOT / "examples/tool_calling"
SPEC = importlib.util.spec_from_file_location(
    "example_arithmetic", TOOL_DIRECTORY / "arithmetic.py"
)
assert SPEC is not None and SPEC.loader is not None
ARITHMETIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARITHMETIC)


def load_function(path: Path, name: str, namespace: dict, class_name: str = ""):
    """Load only a reviewed function definition, without example module imports."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class ArithmeticTests(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        examples = {
            "2 + 3 * 4": 14,
            "(2 + 3) * 4": 20,
            "-3 + +2": -1,
            "7 / 2": 3.5,
            "7 // 2": 3,
            "7 % 2": 1,
            "2**10": 1024,
            "4**0.5": 2,
            "2**-3": 0.125,
            "1_000_000_000_000": 1_000_000_000_000,
            " 1.5 + 2.5 ": 4,
        }
        for expression, expected in examples.items():
            with self.subTest(expression=expression):
                self.assertEqual(ARITHMETIC.evaluate_arithmetic(expression), expected)

    def test_rejects_executable_and_non_arithmetic_syntax(self):
        expressions = [
            "__import__('os').getcwd()",
            "(1).__class__",
            "abs(-2)",
            "[1][0]",
            "{'a': 1}",
            "(1, 2)",
            "[x for x in (1,)]",
            "lambda: 1",
            "True",
            "None",
            "1j",
            "1 << 3",
            "1 and 2",
            "1 if True else 2",
            "'1' * 2",
        ]
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                ARITHMETIC.evaluate_arithmetic(expression)

    def test_resource_and_numeric_limits(self):
        def balanced_sum(terms):
            if terms == 1:
                return "1"
            left = terms // 2
            return f"({balanced_sum(left)}+{balanced_sum(terms - left)})"

        expressions = {
            "length": "1" * (ARITHMETIC.MAX_EXPRESSION_LENGTH + 1),
            "syntax": balanced_sum(24),
            "operation": balanced_sum(18),
            "depth": "-" * (ARITHMETIC.MAX_DEPTH + 1) + "1",
            "exponent": "2**13",
            "numeric": "1000000000000 * 2",
        }
        for limit, expression in expressions.items():
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, f"(?i){limit}"),
            ):
                ARITHMETIC.evaluate_arithmetic(expression)

    def test_invalid_or_undefined_arithmetic(self):
        for expression in ["", "1 +", "1 / 0", "1 % 0", "0**-1", "1e309", "(-1)**0.5"]:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                ARITHMETIC.evaluate_arithmetic(expression)
        for expression in [1, None, True]:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                ARITHMETIC.evaluate_arithmetic(expression)

    def test_both_example_tools_use_safe_evaluator(self):
        for filename in [
            "openai_chat_completion_client_with_tools_xlam.py",
            "openai_chat_completion_client_with_tools_xlam_streaming.py",
        ]:
            with self.subTest(filename=filename):
                calculate = load_function(
                    TOOL_DIRECTORY / filename,
                    "calculate_expression",
                    {"evaluate_arithmetic": ARITHMETIC.evaluate_arithmetic},
                )
                self.assertEqual(calculate("(2+3)*4"), "The result of (2+3)*4 is 20")
                self.assertTrue(
                    calculate("__import__('os')").startswith("Could not calculate")
                )

    def test_helper_imports_for_direct_and_module_execution(self):
        for filename in [
            "openai_chat_completion_client_with_tools_xlam.py",
            "openai_chat_completion_client_with_tools_xlam_streaming.py",
        ]:
            tree = ast.parse((TOOL_DIRECTORY / filename).read_text())
            import_block = next(
                node
                for node in tree.body
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "__package__"
            )
            module = ast.Module(body=[import_block], type_ignores=[])
            for package, directory in [
                ("", TOOL_DIRECTORY),
                ("examples.tool_calling", ROOT),
            ]:
                with self.subTest(filename=filename, package=package):
                    namespace = {"__package__": package}
                    with patch.object(sys, "path", [str(directory), *sys.path]):
                        exec(compile(module, filename, "exec"), namespace)
                    self.assertEqual(namespace["evaluate_arithmetic"]("2+3"), 5)


class ProxyAuthenticationLoggingTests(unittest.TestCase):
    def test_failed_authentication_does_not_log_credentials(self):
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code
                self.detail = detail

        logger = logging.getLogger("proxy-auth-test")
        authenticate = load_function(
            ROOT / "examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py",
            "api_key_authenticate",
            {
                "Header": lambda *_: None,
                "HTTPException": HTTPException,
                "logger": logger,
                "os": os,
                "status": SimpleNamespace(
                    HTTP_403_FORBIDDEN=403, HTTP_500_INTERNAL_SERVER_ERROR=500
                ),
            },
            class_name="Proxy",
        )
        with patch.dict(
            os.environ, {"ADMIN_API_KEY": "configured-test-sentinel"}, clear=True
        ):
            with (
                self.assertLogs(logger, level="WARNING") as logs,
                self.assertRaises(HTTPException) as caught,
            ):
                authenticate(None, "submitted-test-sentinel\nlog-injection")
            self.assertEqual(caught.exception.status_code, 403)
            output = "\n".join(logs.output)
            self.assertIn("Unauthorized access attempt", output)
            self.assertNotIn("submitted-test-sentinel", output)
            self.assertNotIn("configured-test-sentinel", output)
            self.assertNotIn("log-injection", output)
            with self.assertNoLogs(logger, level="WARNING"):
                self.assertIsNone(authenticate(None, "configured-test-sentinel"))


if __name__ == "__main__":
    unittest.main()
