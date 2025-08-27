from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun

# 1️⃣ Istanziare il tool di ricerca
search_tool = DuckDuckGoSearchRun()  # ricerca web automatica

# 2️⃣ Creare l'agente
web_agent = Agent(
    role="Esperto in ricerca web",
    goal="Trovare i primi 3 risultati su DuckDuckGo per un argomento specifico",
    backstory="Un agente specializzato nel trovare informazioni affidabili online",
    tools=[search_tool],
    verbose=True
)

# 3️⃣ Definire il task di ricerca
search_task = Task(
    description="Cerca i primi 3 risultati su DuckDuckGo per l'argomento fornito dall'utente.",
    expected_output="Lista dei primi 3 URL o snippet rilevanti.",
    agent=web_agent
)

# 4️⃣ Creare la crew
poem_crew = Crew(
    agents=[web_agent],
    tasks=[search_task],
    verbose=True
)

# 5️⃣ Avviare l'esecuzione
if __name__ == "__main__":
    topic = input("Inserisci l'argomento da cercare: ")
    result = poem_crew.kickoff(inputs={"argomento": topic})

    print("\n=== Risultati della ricerca ===\n")
    print(result)
