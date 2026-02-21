"""Unit tests for math_tools.py - add, subtract, multiply, divide, calculate."""
import os
import sys
import pytest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extensions.tools.math_tools import add, subtract, multiply, divide, calculate, MATH_TOOLS


class TestAdd:
    """Test the add function."""

    def test_add_positive(self):
        result = add(3.0, 5.0)
        assert "8.0" in result

    def test_add_negative(self):
        result = add(-3.0, -5.0)
        assert "-8.0" in result

    def test_add_mixed(self):
        result = add(-10.0, 7.0)
        assert "-3.0" in result

    def test_add_zero(self):
        result = add(0.0, 0.0)
        assert "0.0" in result

    def test_add_large_numbers(self):
        result = add(1e15, 1e15)
        assert "=" in result

    def test_add_decimals(self):
        result = add(1.5, 2.3)
        assert "3.8" in result

    def test_add_returns_string(self):
        result = add(1.0, 2.0)
        assert isinstance(result, str)
        assert "+" in result


class TestSubtract:
    """Test the subtract function."""

    def test_subtract_positive(self):
        result = subtract(10.0, 3.0)
        assert "7.0" in result

    def test_subtract_negative_result(self):
        result = subtract(3.0, 10.0)
        assert "-7.0" in result

    def test_subtract_zero(self):
        result = subtract(5.0, 0.0)
        assert "5.0" in result

    def test_subtract_same(self):
        result = subtract(5.0, 5.0)
        assert "0.0" in result

    def test_subtract_returns_string(self):
        result = subtract(1.0, 2.0)
        assert isinstance(result, str)
        assert "-" in result


class TestMultiply:
    """Test the multiply function."""

    def test_multiply_positive(self):
        result = multiply(3.0, 5.0)
        assert "15.0" in result

    def test_multiply_by_zero(self):
        result = multiply(100.0, 0.0)
        assert "0.0" in result

    def test_multiply_negative(self):
        result = multiply(-3.0, 5.0)
        assert "-15.0" in result

    def test_multiply_both_negative(self):
        result = multiply(-3.0, -5.0)
        assert "15.0" in result

    def test_multiply_returns_string(self):
        result = multiply(2.0, 3.0)
        assert isinstance(result, str)
        assert "*" in result


class TestDivide:
    """Test the divide function."""

    def test_divide_positive(self):
        result = divide(10.0, 2.0)
        assert "5.0" in result

    def test_divide_by_zero(self):
        result = divide(10.0, 0.0)
        assert "[ERROR]" in result
        assert "zero" in result.lower()

    def test_divide_negative(self):
        result = divide(-10.0, 2.0)
        assert "-5.0" in result

    def test_divide_fraction(self):
        result = divide(1.0, 3.0)
        assert "0.333" in result

    def test_divide_by_one(self):
        result = divide(7.0, 1.0)
        assert "7.0" in result

    def test_divide_returns_string(self):
        result = divide(10.0, 5.0)
        assert isinstance(result, str)
        assert "/" in result


class TestCalculate:
    """Test the calculate function with expressions."""

    def test_simple_addition(self):
        result = calculate("2 + 3")
        assert "5" in result

    def test_complex_expression(self):
        result = calculate("(2 + 3) * 4")
        assert "20" in result

    def test_sqrt(self):
        result = calculate("sqrt(16)")
        assert "4" in result

    def test_pi(self):
        result = calculate("pi")
        assert "3.14" in result

    def test_sin(self):
        result = calculate("sin(0)")
        assert "0" in result

    def test_cos(self):
        result = calculate("cos(0)")
        assert "1" in result

    def test_log(self):
        result = calculate("log(1)")
        assert "0" in result

    def test_exp(self):
        result = calculate("exp(0)")
        assert "1" in result

    def test_abs(self):
        result = calculate("abs(-5)")
        assert "5" in result

    def test_pow(self):
        result = calculate("pow(2, 10)")
        assert "1024" in result

    def test_floor(self):
        result = calculate("floor(3.7)")
        assert "3" in result

    def test_ceil(self):
        result = calculate("ceil(3.2)")
        assert "4" in result

    def test_invalid_expression(self):
        result = calculate("invalid_func(5)")
        assert "[ERROR]" in result

    def test_syntax_error(self):
        result = calculate("2 +* 3")
        assert "[ERROR]" in result

    def test_no_builtins_access(self):
        """Ensure __builtins__ are blocked (security)."""
        result = calculate("__import__('os').system('echo hacked')")
        assert "[ERROR]" in result

    def test_returns_string(self):
        result = calculate("1 + 1")
        assert isinstance(result, str)
        assert "=" in result


class TestMathTools:
    """Test the MATH_TOOLS list."""

    def test_math_tools_count(self):
        assert len(MATH_TOOLS) == 5

    def test_math_tools_names(self):
        names = {t.name for t in MATH_TOOLS}
        assert names == {"add", "subtract", "multiply", "divide", "calculate"}

    def test_all_tools_have_descriptions(self):
        for tool in MATH_TOOLS:
            assert len(tool.description) > 0

    def test_all_tools_callable(self):
        for tool in MATH_TOOLS:
            assert callable(tool.func)
