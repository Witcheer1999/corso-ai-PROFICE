from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from search_exercise_flow.tools.custom_tool import add_numbers

@CrewBase
class SumCrew():
    """Crew per eseguire operazioni matematiche di somma"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def math_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['math_specialist'],
            tools=[add_numbers],
            verbose=True
        )
    
    @task
    def calculate_sum_task(self) -> Task:
        return Task(
            config=self.tasks_config['calculate_sum_task']
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the sum crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )