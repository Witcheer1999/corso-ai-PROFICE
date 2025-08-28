from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen, router
from search_exercise_flow.crews.search_exercise_crew.search_exercise_crew import SearchExerciseCrew
from search_exercise_flow.crews.sum_crew.sum_crew import SumCrew 

class AppState(BaseModel):
    user_choice: str = ""
    user_query: str = ""
    num1: float = 0
    num2: float = 0
    summary: str = ""
    result: str = ""

class MainFlow(Flow[AppState]):

    @start()
    def get_user_choice(self):
        print("Cosa vuoi fare?")
        
        choice = input("📝 Inserisci la tua scelta (ricerca/somma): ").strip().lower()
        
        if choice not in ["ricerca", "somma"]:
            print("⚠️ Scelta non valida. Uso 'ricerca' come default.")
            choice = "ricerca"
        
        self.state.user_choice = choice
        print(f"➡️ Hai scelto: {choice}")
        return choice

    @router(get_user_choice)
    def route_to_function(self):
        if self.state.user_choice == "ricerca":
            return "search_path"
        else:
            return "sum_path"

    # ========== PERCORSO RICERCA ==========
    @listen("search_path")
    def get_search_query(self):
        query = input("📝 Inserisci il tuo argomento di ricerca: ").strip()
        if not query:
            query = "Ultime innovazioni nell'intelligenza artificiale 2024"
            print(f"➡️ Query di esempio: {query}")
        self.state.user_query = query
        return query

    @listen(get_search_query)
    def search_and_summarize(self, query: str):
        print("\n🌐 Eseguo la ricerca con DuckDuckGo...")
        crew = SearchExerciseCrew().crew()
        result = crew.kickoff(inputs={"query": query})
        self.state.summary = str(result)
        return self.state.summary

    @listen(search_and_summarize)
    def display_search_results(self, summary: str):
        print("\n📋 RISULTATI DELLA RICERCA WEB\n")
        print(f"🔍 Query: {self.state.user_query}\n")
        print(summary)
        self.state.result = "Ricerca completata!"
        return self.state.result

    # ========== PERCORSO SOMMA ==========
    @listen("sum_path")
    def get_numbers(self):
        try:
            num1 = float(input("📝 Inserisci il primo numero: "))
            num2 = float(input("📝 Inserisci il secondo numero: "))
        except ValueError:
            print("⚠️ Input non valido. Uso numeri di esempio: 10 e 5")
            num1, num2 = 10, 5
        
        self.state.num1 = num1
        self.state.num2 = num2
        print(f"➡️ Numeri inseriti: {num1} e {num2}")
        return {"num1": num1, "num2": num2}

    @listen(get_numbers)
    def calculate_sum(self, numbers: dict):
        print("\n🧮 Eseguo il calcolo...")
        crew = SumCrew().crew()
        result = crew.kickoff(inputs=numbers)
        self.state.summary = str(result)
        return self.state.summary

    @listen(calculate_sum)
    def display_sum_results(self, summary: str):
        print("\n📊 RISULTATO DEL CALCOLO\n")
        print(f"🔢 Operazione: {self.state.num1} + {self.state.num2}")
        print(summary)
        self.state.result = "Calcolo completato!"
        return self.state.result


def kickoff():
    MainFlow().kickoff()

def plot():
    flow = MainFlow()
    flow.plot("main_flow_plot")

if __name__ == "__main__":
    kickoff()