from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from ragflow.tools.math_tool import execute_math_function  # Import del nostro tool

@CrewBase
class MathCrew():
    """Crew per risolvere problemi matematici"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def math_solver(self) -> Agent:
        return Agent(
            config=self.agents_config['math_solver'],
            tools=[execute_math_function],  # Aggiungiamo il tool
            verbose=True
        )
    
    @task  
    def solve_math_problem_task(self) -> Task:
        return Task(
            config=self.tasks_config['solve_math_problem_task']
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the math solver crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )