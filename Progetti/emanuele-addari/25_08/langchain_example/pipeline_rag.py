from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

# Import RAGAS using absolute imports to avoid name conflicts with local files
import ragas
from ragas import EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

# =========================
# Configurazione
# =========================

load_dotenv()

@dataclass
class Settings:
    # Persistenza FAISS - usa un percorso relativo allo script
    persist_dir: str = None  # Sarà impostato a runtime
    # Text splitting
    chunk_size: int = 500  # Ridotto per catturare meglio domande e risposte
    chunk_overlap: int = 100  # Ridotto per evitare duplicazioni
    # Retriever (MMR)
    search_type: str = "similarity"  # Ricerca per similarità
    k: int = 5  # Numero di documenti da recuperare
    fetch_k: int = 10      # candidati iniziali (per MMR)
    mmr_lambda: float = 0.5  # Bilanciato tra diversità e rilevanza
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Azure OpenAI deployment
    azure_deployment_env: str = "AZURE_INFERENCE_CHAT_DEPLOYMENT_NAME"



SETTINGS = Settings()
# Imposta il percorso assoluto per il vectorstore
SETTINGS.persist_dir = str(Path(__file__).resolve().parent / "faiss_index_example")

def get_embeddings(settings: Settings):
    """
    Restituisce un modello di embedding da Azure OpenAI.
    """
    from langchain_openai import AzureOpenAIEmbeddings
    
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL")
    deployment_name = os.getenv("AZURE_INFERENCE_DEPLOYMENT_NAME")
    api_version = "2023-12-01-preview"

    if not endpoint or not api_key or not deployment_name:
        raise RuntimeError(
            "AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_CREDENTIAL e AZURE_INFERENCE_DEPLOYMENT_NAME devono essere impostate."
        )
    
    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment=deployment_name,
        api_version=api_version,
    )


def get_llm_from_azure(settings: Settings):
    """
    Inizializza un ChatModel puntando ad Azure OpenAI.
    Richiede:
      - AZURE_INFERENCE_ENDPOINT
      - AZURE_INFERENCE_CREDENTIAL
      - AZURE_INFERENCE_CHAT_DEPLOYMENT_NAME
    """
    from langchain_openai import AzureChatOpenAI
    
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL")
    deployment_name = os.getenv("AZURE_INFERENCE_CHAT_DEPLOYMENT_NAME")
    api_version = "2023-12-01-preview"

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        api_version=api_version
    )

def corpus():
    """
    Carica documenti dalla cartella corpus.
    """
    # Usa un percorso assoluto per la cartella corpus
    current_script_path = Path(__file__).resolve().parent
    docs_folder = current_script_path / "corpus"
    
    docs = []
    for file_path in docs_folder.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            doc = Document(
                page_content=content,
                metadata={"id": file_path.stem, "source": file_path.name}
            )
            docs.append(doc)
    
    if not docs:
        raise ValueError(f"Nessun documento markdown trovato nella cartella {docs_folder}")
    
    return docs


def simulate_corpus() -> List[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "---",
            "## ",
            "\n\n", 
            "\n", 
            ". ", 
            "? ", 
            "! ", 
            "; ", 
            ": ",
            ", ", 
            " ", 
            ""
        ],
    )
    chunks = splitter.split_documents(docs)
    print(chunks)
    return chunks


def build_faiss_vectorstore(chunks: List[Document], embeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    Forza sempre la ricostruzione dell'indice per usare i dati più recenti.
    """
    if not docs:
        raise ValueError("Nessun documento fornito per la costruzione del vectorstore")
    
    persist_path = Path(settings.persist_dir)

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


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
        "Sei un assistente esperto. Rispondi in italiano. "
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
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

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


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = retriever.invoke(question)[:k]
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
    Ogni riga contiene: user_input, retrieved_contexts, response, (opzionale) reference.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)

        row = {
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset

def main():
    settings = SETTINGS

    embeddings = get_embeddings(settings)
    llm = get_llm_from_azure(settings)

    # Usa solo i documenti dalla cartella corpus
    docs = corpus()
    
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    retriever = make_retriever(vector_store, settings)

    chain = build_rag_chain(llm, retriever)

    questions = [
        "Qual è la capitale d'Italia?",
        "Quanti minuti ci sono in un'ora?",
        "Chi ha scritto la \"Divina Commedia\"?",
        "Qual è la formula chimica dell'acqua?",
        "In che anno è avvenuta la scoperta dell'America?"
    ]

    # (opzionale) ground truth sintetica per correctness
    ground_truth = {
        questions[0]: "Milano",
        questions[1]: "60",
        questions[2]: "Dante Alighieri",
        questions[3]: "H2O",
        questions[4]: "1492",
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
    # Aggiungi correctness solo se tutte le righe hanno reference
    if all("reference" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = ragas.evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,                 # passa l'istanza LangChain del tuo LLM
        embeddings=embeddings,   # riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    
    # Select only the most important columns for display
    display_cols = ["user_input", "response", "context_precision", "context_recall", 
                   "faithfulness", "answer_relevancy"]
    if "answer_correctness" in df.columns:
        display_cols.append("answer_correctness")
    
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    # Format the display for better readability
    pd_display = df[display_cols].copy()
    # Truncate long text columns
    for col in ["user_input", "response"]:
        pd_display[col] = pd_display[col].str.slice(0, 50) + "..."
    
    print(pd_display.round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")

if __name__ == "__main__":
    main()