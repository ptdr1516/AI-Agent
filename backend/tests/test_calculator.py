"""Tests for the mathematical calculator tool."""
import pytest
from agent.tools.calculator import calculator_tool, CalculatorInput

def test_calculator_valid_math():
    """Test valid mathematical expressions"""
    result = calculator_tool.invoke({"expression": "2 + 2 * 3"})
    assert result == "8"
    
    result = calculator_tool.invoke({"expression": "100 / 5"})
    assert result == "20.0"

def test_calculator_division_by_zero():
    """Test safe handling of division by zero"""
    result = calculator_tool.invoke({"expression": "10 / 0"})
    assert "Error" in result
    assert "division by zero" in result

def test_calculator_invalid_syntax():
    """Test safe handling of random text or invalid python expressions"""
    result = calculator_tool.invoke({"expression": "import os; os.system('echo hacked')"})
    assert "Error" in result
    
    result = calculator_tool.invoke({"expression": "what is 2 + 2?"})
    assert "Error" in result

def test_calculator_pydantic_schema():
    """Ensure the schema correctly processes input string"""
    schema = CalculatorInput(expression="5 * 5")
    assert schema.expression == "5 * 5"
