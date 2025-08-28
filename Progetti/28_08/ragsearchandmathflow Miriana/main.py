from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen, router
from ragflow.crews.search_crew.search_crew import SearchCrew
from ragflow.crews.rag_crew.rag_crew import RagCrew
from crewai import Crew
from ragflow.crews.classifier_crew.classifier_crew import ClassifierCrew
from ragflow.crews.math_crew.math_crew import MathCrew

class IntelligentSearchState(BaseModel):
    user_query: str = ""
    is_math: str = ""  # "MATH" o "NON-MATH"
    domain_classification: str = ""  # "medical" o "general" (solo per non-math)
    search_type: str = ""  # "math", "rag", "web"
    summary: str = ""
    result: str = ""

class IntelligentSearchFlow(Flow[IntelligentSearchState]):

    @start()
    def get_user_question(self):
        print("\n🤖 ASSISTENTE RICERCA INTELLIGENTE")
        print("Fai una domanda e il sistema sceglierà automaticamente la fonte migliore:")
        print("🧮 Domande matematiche → Calcolatore matematico")
        print("🏥 Domande mediche → Database locale (RAG)")
        print("🌍 Domande generali → Ricerca web")
        print()
        
        query = input("❓ Inserisci la tua domanda: ").strip()
        if not query:
            query = "Quali sono i sintomi dell'influenza?"
            print(f"➡️ Domanda di esempio: {query}")
        
        self.state.user_query = query
        return query

    @listen(get_user_question)
    def detect_math_question(self, query: str):
        print("\n🎯 LIVELLO 1: Verifico se è una domanda matematica...")
        
        # Usa ClassifierCrew con task specifico per math detection
        classifier_crew = ClassifierCrew()
        
        # Esegui SOLO il task di math detection usando l'agent math_detector
        math_detector_agent = classifier_crew.math_detector()
        detect_task = classifier_crew.detect_math_task()
        detect_task.agent = math_detector_agent
        
        # Kickoff solo il task di math detection
        temp_crew = Crew(
            agents=[math_detector_agent],
            tasks=[detect_task]
        )
        
        classification_result = temp_crew.kickoff(inputs={"question": query})
        math_classification = str(classification_result).strip().upper()
        
        # Pulisci la risposta per assicurarsi che sia solo MATH o NON-MATH
        if "MATH" in math_classification and "NON" not in math_classification:
            self.state.is_math = "MATH"
            self.state.search_type = "math"
            print("🧮 Domanda MATEMATICA rilevata → Uso calcolatore matematico")
        else:
            self.state.is_math = "NON-MATH"
            print("📝 Domanda NON-MATEMATICA → Procedo con classificazione dominio...")
        
        return self.state.is_math
    
    @router(detect_math_question)
    def route_after_math_detection(self):
        if self.state.is_math == "MATH":
            return "perform_math_calculation"
        else:
            return "classify_nonmath_domain"

    @listen("classify_nonmath_domain")
    def classify_domain_for_nonmath(self):
        print("\n🎯 LIVELLO 2: Classifico il dominio (Medical vs General)...")
        
        # Usa ClassifierCrew con task specifico per domain classification
        classifier_crew = ClassifierCrew()
        
        # Esegui SOLO il task di domain classification usando l'agent domain_classifier
        domain_classifier_agent = classifier_crew.domain_classifier()
        classify_task = classifier_crew.classify_nonmath_domain_task()
        classify_task.agent = domain_classifier_agent
        
        # Kickoff solo il task di domain classification
        temp_crew = Crew(
            agents=[domain_classifier_agent],
            tasks=[classify_task]
        )
        
        classification_result = temp_crew.kickoff(inputs={"question": self.state.user_query})
        domain_classification = str(classification_result).strip().upper()
        
        if "MEDICAL" in domain_classification:
            self.state.search_type = "rag"
            self.state.domain_classification = "medical"
            print("🏥 Domanda MEDICA rilevata → Uso database medico locale")
        else:
            self.state.search_type = "web"
            self.state.domain_classification = "general"
            print("🌍 Domanda GENERALE rilevata → Uso ricerca web")
        
        return self.state.search_type

    @router(classify_domain_for_nonmath)
    def route_after_domain_classification(self):
        if self.state.search_type == "rag":
            return "perform_rag_search"
        else:
            return "perform_web_search"

    # ========== PERCORSO MATEMATICO ==========
    @listen("perform_math_calculation")
    def calculate_with_math(self):
        print("\n🧮 Risolvo il problema matematico...")
        
        math_crew = MathCrew().crew()
        result = math_crew.kickoff(inputs={"question": self.state.user_query})
        self.state.summary = str(result)
        return self.state.summary

    @listen(calculate_with_math)
    def display_math_results(self, summary: str):
        print("\n" + "="*60)
        print("🧮 RISULTATO MATEMATICO")
        print("="*60)
        print(f"❓ Problema: {self.state.user_query}")
        print(f"⚡ Fonte: Calcolatore matematico")
        print("-"*60)
        print(f"📊 Soluzione:\n{summary}")
        print("="*60)
        
        self.state.result = "Calcolo matematico completato!"
        return self.state.result

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
        print("\n🌍 Cerco su internet con DuckDuckGo...")
        
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
        print(f"🌍 Fonte: Ricerca internet")
        print("-"*60)
        print(f"📄 Risultati:\n{summary}")
        print("="*60)
        
        self.state.result = "Ricerca web completata!"
        return self.state.result


def kickoff():
    IntelligentSearchFlow().kickoff()

def plot():
    flow = IntelligentSearchFlow()
    flow.plot("intelligent_search_flow_plot")

if __name__ == "__main__":
    kickoff()