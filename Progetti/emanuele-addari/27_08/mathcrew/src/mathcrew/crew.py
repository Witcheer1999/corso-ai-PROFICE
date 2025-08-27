from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool
)
from mathcrew.tools.custom_tool import SumTool

search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()
sum_tool = SumTool()

@CrewBase
class Mathcrew():
    """Mathcrew crew"""
    topic: str
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def worker(self) -> Agent:
        return Agent(
            config=self.agents_config['worker'],
            verbose=True,
            tools=[search_tool, web_rag_tool, sum_tool]
        )
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            input_variables=["topic"]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Mathcrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
