from typing import Type
from ddgs import DDGS
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field


@tool("DuckDuckGo Search")
def duck_duck_search(query: str) -> str:
    """Searches DuckDuckGo for the given query and returns the results."""
    with DDGS(verify=False) as ddgs:
        results = ddgs._search(query=query, 
                               category='text',
                               region='it', 
                               max_results=3, 
                               safesearch='on'
                               )
        return results

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
