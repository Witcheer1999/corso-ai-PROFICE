from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.tools import tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ResearchCrew():
    """ResearchCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @tool
    def cerca_ddg(topic: str, n: int = 3) -> list[dict]:
        """Cerca su DuckDuckGo e restituisce i primi n risultati come lista di dict."""
        with DDGS(verify=False) as ddgs:
            return [
                {
                    "title": r.get("title"),
                    "snippet": r.get("body")
                }
                for r in ddgs.text(topic, region="it-it", safesearch="off", max_results=n)
            ]
    custom_tool = cerca_ddg

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['ricercatore_web'], # type: ignore[index]
            verbose=True,
            tools=[custom_tool]
        )

    @agent
    def summarize_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['sintetizzatore_contenuti'], # type: ignore[index]
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_task'], # type: ignore[index]
            output_file='output/ricerca.txt'
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['synth_task'], # type: ignore[index]
            output_file='output/risultato.txt'
        )
    
    
    @crew
    def crew(self) -> Crew:
        """Creates the Research crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )