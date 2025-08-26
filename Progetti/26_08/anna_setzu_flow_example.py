from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel
from crewai import LLM

class CityState(BaseModel):
    place: str = ""
    kind: str = ""
    result: str = ""

class CityStateFlow(Flow[CityState]):

    @start()
    def start_method(self):
        """Chiede all'LLM di scrivere SOLO una città o uno stato/paese."""
        llm = LLM(model="azure/gpt-4.1")
        prompt = (
            "Scrivi SOLO il nome di una città oppure di uno stato (paese). "
            "Una sola riga, nessuna spiegazione, nessuna punteggiatura extra. "
            "Rispondi in italiano."
        )
        resp = llm.call(prompt)              
        self.state.place = resp.strip()
        print(f"[start] place = {self.state.place}")

    @router(start_method)
    def branch(self):
        llm = LLM(model="azure/gpt-4.1")
        prompt = (
            f"Considera il nome '{self.state.place}'. È una città oppure uno stato/paese? "
            "Rispondi con UNA SOLA parola in inglese: city oppure country."
        )
        kind_raw = llm.call(prompt).strip().lower()   
        self.state.kind = "city" if "city" in kind_raw else "country"
        return self.state.kind                        

    @listen("city")                                   
    def city_node(self):
        llm = LLM(model="azure/gpt-4.1")
        prompt = (
            f"Dammi un fatto interessante e conciso su {self.state.place}. "
            "Massimo una frase, senza preamboli. Rispondi in italiano."
        )
        fact = llm.call(prompt).strip()               # <-- STRINGA
        self.state.result = f"Fatto su {self.state.place}: {fact}"
        print(f"[city_node] {self.state.result}")

    @listen("country")                                
    def state_node(self):
        llm = LLM(model="azure/gpt-4.1")
        prompt = (
            f"Elenca i paesi confinanti con {self.state.place} (solo confini terrestri). "
            "Rispondi con un elenco separato da virgole, senza spiegazioni. "
            f"Se {self.state.place} non ha confini terrestri, scrivi esattamente: Nessun confine terrestre. "
            "Rispondi in italiano."
        )
        neighbors = llm.call(prompt).strip()          
        self.state.result = f"Paesi confinanti con {self.state.place}: {neighbors}"
        print(f"[state_node] {self.state.result}")

def kickoff():
    flow = CityStateFlow()
    flow.plot("my_flow_plot")
    return flow.kickoff()

if __name__ == "__main__":
    kickoff()
