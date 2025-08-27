# src/research_crew/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from web_search_crew.tools.custom_tool import duck_duck_search


duck_duck_tool = duck_duck_search

@CrewBase
class WebSearchCrew():
    """Research crew for comprehensive topic analysis and reporting"""

    agents: List[BaseAgent]
    tasks: List[Task]

    
    @agent
    def web_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config['web_searcher'], # type: ignore[index]
            verbose=True,
            tools=[duck_duck_tool]
        )


    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'], # type: ignore[index]
            verbose=True
        )


    @task
    def web_search_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_search_task'] # type: ignore[index]
        )


    @task
    def summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config['summarization_task'], # type: ignore[index]
            output_file='output/report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the web search crew pipeline"""
        return Crew(
            agents=[
                self.web_searcher(),
                self.summarizer()
            ],
            tasks=[
                self.web_search_task(),
                self.summarization_task()
            ],
            process=Process.sequential,
            verbose=True,
        )