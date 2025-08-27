from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    first_number: float = Field(..., description="First number to add.")
    second_number: float = Field(..., description="Second number to add.")

class SumTool(BaseTool):
    name: str = "SumTool"
    description: str = ("Tool for summing numbers. Only tool to use when the user asks to sum numbers")
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, first_number: float, second_number: float) -> str:
        # Implementation goes here
        result = first_number + second_number
        return str(result)
