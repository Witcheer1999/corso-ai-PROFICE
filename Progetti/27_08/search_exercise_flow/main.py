from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen
from search_exercise_flow.crews.search_exercise_crew.search_exercise_crew import SearchExerciseCrew

class WebSearchState(BaseModel):
    user_query: str = ""
    summary: str = ""

class WebSearchFlow(Flow[WebSearchState]):

    @start()
    def get_user_input(self):
        print("\n🔍 WEB SEARCH FLOW")
        query = input("📝 Inserisci il tuo argomento di ricerca: ").strip()
        if not query:
            query = "Ultime innovazioni nell'intelligenza artificiale 2024"
            print(f"➡️ Query di esempio: {query}")
        self.state.user_query = query
        return query

    @listen(get_user_input)
    def search_and_summarize(self, query: str):
        print("\n🌐 Eseguo la ricerca con DuckDuckGo...")
        crew = SearchExerciseCrew().crew()
        result = crew.kickoff(inputs={"query": query})
        self.state.summary = str(result)
        return self.state.summary

    @listen(search_and_summarize)
    def display_results(self, summary: str):
        print("\n📋 RISULTATI DELLA RICERCA WEB\n")
        print(f"🔍 Query: {self.state.user_query}\n")
        print(summary)
        return "Flow completato!"


def kickoff():
    WebSearchFlow().kickoff()

def plot():
    flow = WebSearchFlow()
    flow.plot("search_exercise_plot")

if __name__ == "__main__":
    kickoff()