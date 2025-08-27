import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione Azure OpenAI LLM - PARAMETRI CORRETTI
azure_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    openai_api_key=os.getenv("AZURE_API_KEY"),  # Cambiato da api_key
    openai_api_version=os.getenv("AZURE_API_VERSION"),  # Cambiato da api_version
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME")
)

# Inizializza DuckDuckGo search
ddg_search = DuckDuckGoSearchRun()

# Crea un tool personalizzato per CrewAI usando DuckDuckGo
@tool
def web_search_tool(query: str) -> str:
    """
    Effettua una ricerca web utilizzando DuckDuckGo.
    
    Args:
        query: La query di ricerca da eseguire
        
    Returns:
        I risultati della ricerca web
    """
    try:
        results = ddg_search.run(query)
        if not results:
            return "Nessun risultato trovato per la query specificata."
        return results
    except Exception as e:
        return f"Errore durante la ricerca: {str(e)}"

# Definizione dell'agente di ricerca web
web_researcher = Agent(
    role="Web Research Specialist",
    goal="Condurre ricerche approfondite sul web per trovare informazioni accurate e aggiornate",
    backstory="""Sei un esperto ricercatore web con anni di esperienza nel trovare 
    e analizzare informazioni online. Hai un talento particolare nel distinguere 
    fonti affidabili da quelle meno attendibili e sei capace di sintetizzare 
    informazioni complesse in report chiari e concisi. Utilizzi DuckDuckGo come 
    motore di ricerca principale per garantire privacy e risultati neutrali.""",
    tools=[web_search_tool],
    llm=azure_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3  # Limita le iterazioni per evitare loop infiniti
)

# Funzione per creare un task di ricerca
def create_research_task(query):
    """Crea un task di ricerca personalizzato basato sulla query fornita"""
    return Task(
        description=f"""Effettua una ricerca dettagliata su: {query}
        
        Il tuo compito include:
        1. Utilizzare il web_search_tool per cercare informazioni rilevanti
        2. Effettuare ricerche multiple con termini diversi se necessario
        3. Analizzare e verificare le informazioni trovate
        4. Riassumere i punti chiave trovati
        5. Fornire un report strutturato con i risultati principali
        
        IMPORTANTE: Usa il web_search_tool fornito per effettuare le ricerche.
        """,
        expected_output="""Un report dettagliato che includa:
        - Sintesi delle informazioni trovate
        - Punti chiave e insights principali
        - Informazioni sulle fonti consultate
        - Eventuali raccomandazioni o conclusioni
        - Data di aggiornamento delle informazioni (se disponibile)""",
        agent=web_researcher
    )

# Funzione principale per eseguire una ricerca
def run_web_research(query):
    """Esegue una ricerca web utilizzando CrewAI con DuckDuckGo"""
    
    print(f"\n🔎 Preparazione ricerca con DuckDuckGo...")
    
    # Crea il task di ricerca
    research_task = create_research_task(query)
    
    # Crea il crew con l'agente e il task
    crew = Crew(
        agents=[web_researcher],
        tasks=[research_task],
        verbose=True
    )
    
    print(f"🚀 Avvio del crew di ricerca...\n")
    
    # Esegui il crew e ottieni i risultati
    result = crew.kickoff()
    
    return result

# Test diretto del tool di ricerca
def test_search_tool(query):
    """Funzione di test per verificare che il tool funzioni correttamente"""
    print(f"\n🧪 Test diretto del tool di ricerca...")
    print(f"Query: {query}")
    # Usa .run() per eseguire il tool invece di chiamarlo direttamente
    result = web_search_tool.run(query)
    print(f"Risultato: {result[:500]}..." if len(result) > 500 else f"Risultato: {result}")
    return result

# Esempio di utilizzo
if __name__ == "__main__":
    print("=" * 60)
    print("CREWAI WEB RESEARCH FLOW (con DuckDuckGo)")
    print("=" * 60)
    print("✅ Nessuna API key aggiuntiva richiesta!")
    print("✅ Utilizza solo le credenziali Azure OpenAI dal .env")
    
    # Richiedi input all'utente o usa una query di esempio
    user_query = input("\nInserisci il tuo argomento di ricerca (o premi Enter per usare l'esempio): ")
    
    if not user_query:
        user_query = "Ultime innovazioni nell'intelligenza artificiale nel 2024"
        print(f"Usando query di esempio: {user_query}")
    
    # Opzione per testare prima il tool
    test_option = input("\nVuoi testare prima il tool di ricerca? (s/n, default: n): ")
    if test_option.lower() == 's':
        test_search_tool(user_query)
        proceed = input("\nProcedere con la ricerca completa CrewAI? (s/n): ")
        if proceed.lower() != 's':
            print("Ricerca annullata.")
            exit()
    
    print(f"\n🔍 Avvio ricerca su: {user_query}")
    print("-" * 60)
    
    try:
        # Esegui la ricerca
        results = run_web_research(user_query)
        
        print("\n" + "=" * 60)
        print("📊 RISULTATI DELLA RICERCA")
        print("=" * 60)
        print(results)
        
        # Salva i risultati in un file (opzionale)
        save_option = input("\n\nVuoi salvare i risultati in un file? (s/n, default: n): ")
        if save_option.lower() == 's':
            filename = f"ricerca_{user_query[:30].replace(' ', '_')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {user_query}\n")
                f.write("=" * 60 + "\n")
                f.write(str(results))
            print(f"✅ Risultati salvati in: {filename}")
        
    except Exception as e:
        print(f"\n❌ Errore durante la ricerca: {str(e)}")
        print("\nAssicurati di avere:")
        print("1. Le credenziali Azure OpenAI corrette nel file .env")
        print("2. Le dipendenze installate:")
        print("   pip install crewai crewai-tools python-dotenv langchain-openai langchain-community duckduckgo-search")
        print("\nDettagli errore completo:")
        import traceback
        traceback.print_exc()