from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen, router
from ragflow.crews.search_crew.search_crew import SearchCrew
from ragflow.crews.rag_crew.rag_crew import RagCrew
from ragflow.crews.classifier_crew.classifier_crew import ClassifierCrew

class IntelligentSearchState(BaseModel):
    user_query: str = ""
    domain_classification: str = ""  # "medical", "general"
    search_type: str = ""  # "rag" o "web"
    summary: str = ""
    result: str = ""

class IntelligentSearchFlow(Flow[IntelligentSearchState]):

    @start()
    def get_user_question(self):
        print("\n🤖 ASSISTENTE RICERCA INTELLIGENTE")
        print("Fai una domanda e il sistema sceglierà automaticamente la fonte migliore:")
        print("🏥 Domande mediche → Database locale (RAG)")
        print("🌐 Domande generali → Ricerca web")
        print()
        
        query = input("❓ Inserisci la tua domanda: ").strip()
        if not query:
            query = "Quali sono i sintomi dell'influenza?"
            print(f"➡️ Domanda di esempio: {query}")
        
        self.state.user_query = query
        return query

    @listen(get_user_question)
    def classify_question_domain(self, query: str):
        print("\n🧠 Analizzo il dominio della domanda...")
        
        # Usa la ClassifierCrew per determinare se è medicina o altro
        classifier_crew = ClassifierCrew().crew()
        classification_result = classifier_crew.kickoff(inputs={"question": query})
        
        # Estrai la classificazione dal risultato
        classification = str(classification_result).strip().upper()
        
        if "MEDICAL" in classification:
            self.state.search_type = "rag"
            self.state.domain_classification = "medical"
            print("🏥 Domanda medica rilevata → Uso database medico locale")
        else:
            self.state.search_type = "web"
            self.state.domain_classification = "general"
            print("🌐 Domanda generale rilevata → Uso ricerca web")
        
        return self.state.search_type

    @router(classify_question_domain)
    def route_search_type(self):
        if self.state.search_type == "rag":
            return "perform_rag_search"
        else:
            return "perform_web_search"

    # ========== PERCORSO RAG MEDICO ==========
    @listen("perform_rag_search")
    def search_with_rag(self):
        print("\n📚 Cerco nel database medico locale...")
        
        rag_crew = RagCrew().crew()
        result = rag_crew.kickoff(inputs={"question": self.state.user_query})
        self.state.summary = str(result)
        return self.state.summary

    @listen(search_with_rag)
    def display_rag_results(self, summary: str):
        print("\n" + "="*60)
        print("📋 RISPOSTA DAL DATABASE MEDICO")
        print("="*60)
        print(f"❓ Domanda: {self.state.user_query}")
        print(f"🏥 Fonte: Database medico locale")
        print("-"*60)
        print(f"📚 Risposta:\n{summary}")
        print("="*60)
        
        self.state.result = "Ricerca medica completata!"
        return self.state.result

    # ========== PERCORSO RICERCA WEB ==========
    @listen("perform_web_search")
    def search_with_web(self):
        print("\n🌐 Cerco su internet con DuckDuckGo...")
        
        search_crew = SearchCrew().crew()
        result = search_crew.kickoff(inputs={"query": self.state.user_query})
        self.state.summary = str(result)
        return self.state.summary

    @listen(search_with_web)
    def display_web_results(self, summary: str):
        print("\n" + "="*60)
        print("📋 RISULTATI DALLA RICERCA WEB")
        print("="*60)
        print(f"❓ Domanda: {self.state.user_query}")
        print(f"🌐 Fonte: Ricerca internet")
        print("-"*60)
        print(f"📄 Risultati:\n{summary}")
        print("="*60)
        
        self.state.result = "Ricerca web completata!"
        return self.state.result


def kickoff():
    IntelligentSearchFlow().kickoff()
    #plot()

def plot():
    flow = IntelligentSearchFlow()
    flow.plot("intelligent_search_flow_plot")

if __name__ == "__main__":
    kickoff()
