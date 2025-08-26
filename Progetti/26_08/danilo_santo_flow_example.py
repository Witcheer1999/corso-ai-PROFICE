#!/usr/bin/env python

from typing import Any, Dict, Literal
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

class LocationState(BaseModel):
    location_name: str = ""
    location_type: Literal["city", "state"] = "city"
    result: str = ""

class LocationInfoFlow(Flow[LocationState]):
    
    @start()
    def generate_location(self) -> str:
        """L'LLM genera casualmente uno stato o una città"""
        
        from crewai import LLM
        
        llm = LLM(model="gpt-4o-mini")
        
        messages = [{
            "role": "user", 
            "content": """Genera SOLO il nome di una location (stato/paese oppure città). 
            Rispondi con una sola parola o massimo due parole.
            
            Esempi:
            - Francia
            - Roma  
            - Germania
            - Tokyo
            - Brasile
            - Parigi
            
            Genera ora una location casuale:"""
        }]
        
        response = llm.call(messages=messages)
        location = response.strip()
        
        print(f"🌍 Location generata: {location}")
        self.state.location_name = location
        
        return location

    @listen(generate_location)
    def classify_location(self, location: str) -> Literal["city", "state"]:
        """Classifica se è una città o uno stato/paese e routing decision"""
        
        from crewai import LLM
        
        llm = LLM(model="gpt-4o-mini")
        
        messages = [{
            "role": "user",
            "content": f"""Dimmi se "{location}" è una CITTÀ o uno STATO/PAESE.
            
            Rispondi SOLO con una di queste due parole:
            - CITY (se è una città)  
            - STATE (se è uno stato, paese, nazione)
            
            Location: {location}
            Risposta:"""
        }]
        
        response = llm.call(messages=messages).strip().lower()
        
        # Normalizza la risposta per il router
        if "city" in response or "città" in response:
            location_type = "city"
            print(f"📍 Classificato come: Città")
        else:
            location_type = "state" 
            print(f"📍 Classificato come: Stato/Paese")
            
        self.state.location_type = location_type
        return location_type

    @router(classify_location)
    def route_by_location_type(self, classification: Literal["city", "state"]) -> Literal["handle_city", "handle_state"]:
        """Router che decide il flusso in base al tipo di location"""
        
        if classification == "city":
            print("🔀 Routing verso: gestione città")
            return "handle_city"
        else:
            print("🔀 Routing verso: gestione stato")
            return "handle_state"

    @listen("handle_city")
    def get_city_fact(self) -> str:
        """Restituisce un fatto interessante sulla città"""
        
        from crewai import LLM
        
        llm = LLM()
        location_name = self.state.location_name
        
        messages = [{
            "role": "user",
            "content": f"""Dimmi un fatto interessante e curioso sulla città di {location_name}.
            
            Rispondi con un fatto specifico, storico, culturale o particolare che molte persone non conoscono.
            Mantieni la risposta breve (2-3 frasi massimo).
            
            Città: {location_name}"""
        }]
        
        response = llm.call(messages=messages)
        
        print(f"🏙️ Fatto interessante su {location_name}:")
        print(f"   {response}")
        
        self.state.result = response
        return response

    @listen("handle_state")
    def get_bordering_countries(self) -> str:
        """Restituisce i paesi confinanti con lo stato"""
        
        from crewai.llm import LLM
        
        llm = LLM(model="gpt-4o-mini")
        location_name = self.state.location_name
        
        messages = [{
            "role": "user", 
            "content": f"""Elenca i paesi che confinano con {location_name}.
            
            Rispondi con la lista dei paesi confinanti, separati da virgole.
            Se è un'isola o non ha confini terrestri, specifica "Non ha confini terrestri" o "È un'isola".
            
            Stato/Paese: {location_name}"""
        }]
        
        response = llm.call(messages=messages)
        
        print(f"🗺️ Paesi confinanti con {location_name}:")
        print(f"   {response}")
        
        self.state.result = response
        return response

    @listen(get_city_fact)
    @listen(get_bordering_countries)
    def finalize_result(self, info: str):
        """Finalizza e mostra il risultato finale"""
        
        print("\n" + "="*60)
        print("🎯 RISULTATO FINALE:")
        print("="*60)
        print(f"Location: {self.state.location_name}")
        print(f"Tipo: {'🏙️ Città' if self.state.location_type == 'city' else '🗺️ Stato/Paese'}")
        print(f"Info: {self.state.result}")
        print("="*60 + "\n")

def kickoff():
    """
    Run the flow.
    """
    flow = LocationInfoFlow()
    flow.kickoff()

def plot():
    """
    Plot the flow.
    """
    flow = LocationInfoFlow()
    flow.plot()

if __name__ == "__main__":
    kickoff()