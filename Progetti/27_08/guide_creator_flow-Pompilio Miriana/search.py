from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen
from crewai import LLM
from crews.search_crew.search_crew import SearchCrew

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
        crew = SearchCrew().crew()
        result = crew.kickoff(inputs={"query": query})
        self.state.summary = str(result)
        return self.state.summary

    @listen(search_and_summarize)
    def display_results(self, summary: str):
        print("\n📋 RISULTATI DELLA RICERCA WEB\n")
        print(f"🔍 Query: {self.state.user_query}\n")
        print(summary)
        return "Flow completato!"


flow = WebSearchFlow()
flow.kickoff()
