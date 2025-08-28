from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from ragflow.tools.rag_tool import search_rag

@CrewBase
class RagCrew():
    """Crew per eseguire ricerche nel RAG medico locale"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def medical_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['medical_specialist'],
            tools=[search_rag],
            verbose=True
        )
    
    @task
    def rag_search_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_search_task']
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the RAG crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )