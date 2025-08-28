from crewai.tools import tool
from ddgs import DDGS

@tool
def search_web(query: str) -> str:
    """
    Effettua una ricerca web utilizzando DuckDuckGo e restituisce i primi 3 risultati.
    """
    try:
        with DDGS(verify=False) as ddgs:
            risultati = list(ddgs.text(query, region="it-it", safesearch="off", max_results=3))
        
        if not risultati:
            return f"Nessun risultato trovato per la query: {query}"
        
        formatted_results = f"Risultati della ricerca per '{query}':\n\n"
        for i, r in enumerate(risultati, 1):
            titolo = r.get("title", "Senza titolo")
            url = r.get("href") or r.get("url") or "URL non disponibile"
            snippet = r.get("body", "Nessuna descrizione disponibile")
            formatted_results += f"RISULTATO {i}:\nTitolo: {titolo}\nURL: {url}\nDescrizione: {snippet}\n{'-'*40}\n"
        
        return formatted_results
    except Exception as e:
        return f"Errore durante la ricerca: {str(e)}"
