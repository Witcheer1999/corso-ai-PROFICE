from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from tools.custom_tool import search_web

@CrewBase
class SearchCrew():
    """Crew per eseguire ricerche web e creare riassunti """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def web_research_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['web_research_specialist'],
            tools=[search_web],
            verbose=True
        )
    
    @task
    def search_and_summarize_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_and_summarize_task']
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )