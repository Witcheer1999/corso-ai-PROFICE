from dotenv import load_dotenv
import os
from dataclasses import dataclass
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from pathlib import Path
from typing import List
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)
load_dotenv()

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    key = os.getenv("API_KEY")
    llm_key = os.getenv("API_LLM_KEY")
    azure_llm_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
    #hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    model_name: str = "gpt-4o"
    lmstudio_model_env: str = "LMSTUDIO_MODEL"  # nome del modello in LM Studio, via env var


def split_documents(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ":", ";", " "]
    )
    chunks = splitter.split_documents(docs)
    return chunks

def define_vector_db(settings:Settings,chunks,client : AzureOpenAIEmbeddings):
    index = FAISS.from_documents(
        documents=chunks,
        #passare la funzione di embedding 
        embedding=client
    )
    return index

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset

def load_real_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Carica documenti reali da file di testo (es. .txt, .md) all'interno di una cartella.
    Ogni file viene letto e convertito in un oggetto Document con metadato 'source'.
    """
    folder = Path(folder_path)
    documents: List[Document] = []

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"La cartella '{folder_path}' non esiste o non è una directory.")

    for file_path in folder.glob("**/*"):
        if file_path.suffix.lower() not in [".txt", ".md"]:
            continue  # ignora file non supportati

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        # Aggiunge il metadato 'source' per citazioni (es. nome del file)
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    return documents


def define_embedder(settings : Settings):
    embedding = AzureOpenAIEmbeddings(
        api_version="2024-12-01-preview",
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.key,
    )
    return embedding

def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)

def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi nella lingua della domanda. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto, anche se la risposta ti sembra errata.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         )#"2) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )
    
def define_llm(settings: Settings):
    llm = AzureChatOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=settings.azure_llm_endpoint,
        api_key=settings.llm_key,
        azure_deployment="gpt-4o",  # Replace with actual deployment
        temperature=0.1,
        max_tokens=1000,  # Optional: control response length
    )
    return llm

def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)

def main():
    settings = Settings()

    # 1) Componenti
    client_embeddings = define_embedder(settings)
    client_llm = define_llm(settings)

    docs = load_real_documents_from_folder("25-08\\es1\\corpus")
    chunks = split_documents(docs)
    vector_store = define_vector_db(settings, chunks,client_embeddings)
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(client_llm, retriever)

    # 5) Esempi di domande
    questions = [
        "Qual è la capitale d'Italia? ",
        "Quanti minuti ci sono in un'ora? ",
        "Chi ha scritto la Divina Commedia?",
        "Qual è la formula chimica dell'acqua?"
    ]

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = rag_answer(q, chain)
        print(ans)
        print()

    questions = [
        "Che cos'è una pipeline RAG e quali sono le sue fasi principali?",
        "A cosa serve FAISS e quali capacità offre?",
        "Cos'è MMR e perché è utile durante il retrieval?",
        "Quale dimensione hanno gli embedding prodotti da all-MiniLM-L6-v2?"
    ]

    # (opzionale) ground truth sintetica per correctness
    ground_truth = {
        questions[0]: "Indicizzazione (caricamento, splitting, embedding, storage) e retrieval + generazione.",
        questions[1]: "Libreria per ricerca di similarità e clustering di vettori densi (ANN/NNN) scalabile.",
        questions[2]: "Bilancia pertinenza e diversità per ridurre ridondanza e coprire aspetti differenti.",
        questions[3]: "384",
    }

    # 6) Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 7) Scegli le metriche
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=client_llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=define_embedder(settings),  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")

# if __name__ == "__main__":
#      main()