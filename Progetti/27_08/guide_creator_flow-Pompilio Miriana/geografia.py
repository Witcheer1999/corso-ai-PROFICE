from crewai.flow import Flow, start, listen, router
from crewai.flow.flow import or_
from crewai import LLM
import os

# Carica variabili d'ambiente
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="azure/gpt-4o",
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION")
)

class GeografiaFlow(Flow):

    @start()
    def genera_localita(self):
        response = llm.call("Scrivi il nome di una città o di uno stato a caso.")
        localita = response.strip() if response else "Città non trovata"
        print(f"Località: {localita}")
        return {"localita": localita}

    @listen("genera_localita")
    def classifica_localita(self, result):
        localita = result["localita"]
        response = llm.call(f"'{localita}' è una città o uno stato? Rispondi solo con 'città' o 'stato'.")
        tipo = response.lower().strip() if response else "città"
        print(f"Tipo: {tipo}")
        return {"tipo": tipo, "localita": localita}

    @listen("classifica_localita")
    def smista(self, result):
        # salva localita nello stato globale
        self.state["localita"] = result["localita"]
        if "città" in result["tipo"]:
            print("→ Branch città")
            return "fatto_citta"
        else:
            print("→ Branch stato")
            return "confini_stato"

    @listen("smista")
    def fatto_citta(self, result):
        if result != "fatto_citta":
            return None
        localita = self.state.get("localita")
        response = llm.call(f"Dimmi un fatto interessante sulla città di {localita}.")
        fatto = response if response else f"{localita} è una città interessante"
        print(f"Fatto: {fatto}")
        self.state["risultato_finale"] = fatto  # salva nel flusso
        return fatto  # opzionale, se vuoi ritorno immediato

    @listen("smista")
    def confini_stato(self, result):
        if result != "confini_stato":
            return None
        localita = self.state.get("localita")
        response = llm.call(f"Quali paesi confinano con {localita}?")
        confini = response if response else f"Confini di {localita} da verificare"
        print(f"Confini: {confini}")
        self.state["risultato_finale"] = confini
        return confini



if __name__ == "__main__":
    print("Configurazione Azure:")
    print(f"API Key: {'OK' if os.getenv('AZURE_API_KEY') else 'MANCANTE'}")
    print(f"Endpoint: {os.getenv('AZURE_API_BASE') or 'MANCANTE'}")
    
    flow = GeografiaFlow()
    flow.plot("flow.png")
    print("Plot salvato in flow.png")
    
    result = flow.kickoff()
    print(f"\nRisultato finale: {flow.state.get('risultato_finale')}")