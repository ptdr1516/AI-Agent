from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from core.logger import log

class CalculatorInput(BaseModel):
    expression: str = Field(description="A strictly mathematical expression containing numbers and operators (+, -, *, /, **, ()). No strings.")

@tool("calculator_tool", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """Use this tool exclusively for executing mathematical calculations. Do not guess mathematics."""
    try:
        # Pre-process common LLM math symbols into Python-evaluable operators
        expression = expression.replace("×", "*").replace("x", "*").replace("^", "**")

        allowed_chars = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed_chars):
            log.warning(f"Invalid math expression blocked: {expression}")
            return "Error: Invalid characters. Expression must be strictly mathematical numbers and operators."

        # Safe eval using constrained subset
        result = eval(expression, {"__builtins__": {}})
        log.info(f"Calculated: {expression} = {result}")
        return str(result)
    except Exception as e:
        log.error(f"Error evaluating '{expression}': {e}")
        return f"Error evaluating expression: {str(e)}"
