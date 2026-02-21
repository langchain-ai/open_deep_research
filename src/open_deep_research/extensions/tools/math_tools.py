"""Basic mathematical operation tools."""
import math
from langchain_core.tools import StructuredTool
from extensions.models.tool_schemas import (
    AddInput, SubtractInput, MultiplyInput, DivideInput, CalculateInput
)


def add(a: float, b: float) -> str:
    """Add two numbers together."""
    result = a + b
    return f"{a} + {b} = {result}"


def subtract(a: float, b: float) -> str:
    """Subtract second number from first."""
    result = a - b
    return f"{a} - {b} = {result}"


def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    result = a * b
    return f"{a} * {b} = {result}"


def divide(a: float, b: float) -> str:
    """Divide first number by second."""
    if b == 0:
        return "[ERROR] Cannot divide by zero"
    result = a / b
    return f"{a} / {b} = {result}"


def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely.
    
    Supports: +, -, *, /, sqrt, sin, cos, tan, pi, e, log, exp, abs, round, pow
    """
    try:
        # Safe evaluation with limited functions
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"[ERROR] Error calculating expression: {str(e)}"


# Create LangChain tools
add_tool = StructuredTool.from_function(
    func=add,
    name="add",
    description="Add two numbers together",
    args_schema=AddInput
)

subtract_tool = StructuredTool.from_function(
    func=subtract,
    name="subtract",
    description="Subtract second number from first number",
    args_schema=SubtractInput
)

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

divide_tool = StructuredTool.from_function(
    func=divide,
    name="divide",
    description="Divide first number by second number",
    args_schema=DivideInput
)

calculate_tool = StructuredTool.from_function(
    func=calculate,
    name="calculate",
    description="Calculate a mathematical expression. Supports: +, -, *, /, sqrt, sin, cos, tan, pi, e, log, exp, abs, round, pow",
    args_schema=CalculateInput
)


# Export all math tools as a list
MATH_TOOLS = [add_tool, subtract_tool, multiply_tool, divide_tool, calculate_tool]


__all__ = ['MATH_TOOLS', 'add', 'subtract', 'multiply', 'divide', 'calculate']
