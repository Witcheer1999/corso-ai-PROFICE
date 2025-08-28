# classifier_crew.py aggiornato
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class ClassifierCrew():
    """Crew per classificare il dominio delle domande utente a due livelli"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def math_detector(self) -> Agent:
        return Agent(
            config=self.agents_config['math_detector'],
            verbose=True
        )
    
    @agent
    def domain_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config['domain_classifier'], 
            verbose=True
        )
    
    @task
    def detect_math_task(self) -> Task:
        return Task(
            config=self.tasks_config['detect_math_task']
        )
    
    @task
    def classify_nonmath_domain_task(self) -> Task:
        return Task(
            config=self.tasks_config['classify_nonmath_domain_task']
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the two-level classifier crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )