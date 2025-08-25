from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv


# =========================
# Configurazione Azure
# =========================

load_dotenv()

@dataclass
class AzureSettings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_azure"
    
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 100
    
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    
    # Azure OpenAI - Embedding Model
    azure_embedding_endpoint: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
    azure_embedding_key: str = os.getenv("AZURE_OPENAI_EMBEDDING_KEY", "")
    azure_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    azure_embedding_api_version: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01")
    
    # Azure OpenAI - Chat Model
    azure_chat_endpoint: str = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT", "")
    azure_chat_key: str = os.getenv("AZURE_OPENAI_CHAT_KEY", "")
    azure_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
    azure_chat_api_version: str = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2024-12-01-preview")

    # Opzionale: temperatura e altri parametri del modello
    temperature: float = 0.0
    max_tokens: int = 1000


SETTINGS = AzureSettings()


# =========================
# Componenti Azure AI
# =========================

def get_azure_embeddings(settings: AzureSettings) -> AzureOpenAIEmbeddings:
    """
    Restituisce un modello di embedding usando Azure OpenAI.
    Richiede un deployment di un modello embedding (es. text-embedding-ada-002) su Azure AI Foundry.
    """
    if not settings.azure_embedding_endpoint or not settings.azure_embedding_key:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT e AZURE_OPENAI_EMBEDDING_KEY devono essere configurate. "
            "Assicurati di averle impostate nel file .env"
        )
    
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_embedding_endpoint,
        api_key=settings.azure_embedding_key,
        azure_deployment=settings.azure_embedding_deployment,
        api_version=settings.azure_embedding_api_version
    )


def get_azure_chat_model(settings: AzureSettings) -> AzureChatOpenAI:
    """
    Inizializza un ChatModel usando Azure OpenAI.
    Richiede un deployment di un modello chat (es. gpt-4, gpt-35-turbo) su Azure AI Foundry.
    """
    if not settings.azure_chat_endpoint or not settings.azure_chat_key:
        raise ValueError(
            "AZURE_OPENAI_CHAT_ENDPOINT e AZURE_OPENAI_CHAT_KEY devono essere configurate. "
            "Assicurati di averle impostate nel file .env"
        )
    
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_chat_endpoint,
        api_key=settings.azure_chat_key,
        azure_deployment=settings.azure_chat_deployment,
        api_version=settings.azure_chat_api_version,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )


def simulate_corpus() -> List[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    (Identico all'originale)
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
        Document(
            page_content=(
                "Azure AI Foundry provides a unified platform for deploying and managing AI models. "
                "It supports various model types including GPT-4, custom models, and embedding models "
                "with enterprise-grade security and scalability."
            ),
            metadata={"id": "doc6", "source": "azure-ai-foundry.md"}
        ),
    ]
    return docs


def split_documents(docs: List[Document], settings: AzureSettings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    (Identico all'originale)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    Usa Azure OpenAI Embeddings per la vettorizzazione.
    """
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: AzureSettings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        print(f"Caricamento indice FAISS esistente da {settings.persist_dir}...")
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print(f"Costruzione nuovo indice FAISS in {settings.persist_dir}...")
    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: AzureSettings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    (Identico all'originale)
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
    (Identico all'originale)
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm: AzureChatOpenAI, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    Usa Azure OpenAI Chat model per la generazione.
    """
    system_prompt = (
        "Sei un assistente esperto basato su Azure AI. Rispondi in italiano. "
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


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)


def validate_azure_configuration(settings: AzureSettings):
    """
    Valida che tutte le configurazioni Azure necessarie siano presenti.
    """
    required_settings = [
        ("AZURE_OPENAI_EMBEDDING_ENDPOINT", settings.azure_embedding_endpoint),
        ("AZURE_OPENAI_EMBEDDING_KEY", settings.azure_embedding_key),
        ("AZURE_OPENAI_CHAT_ENDPOINT", settings.azure_chat_endpoint),
        ("AZURE_OPENAI_CHAT_KEY", settings.azure_chat_key),
    ]
    
    missing = [name for name, value in required_settings if not value]
    
    if missing:
        print("=" * 80)
        print("CONFIGURAZIONE AZURE AI FOUNDRY RICHIESTA:")
        print("-" * 80)
        print("Assicurati di avere le seguenti variabili d'ambiente nel file .env:")
        print()
        for var in missing:
            print(f"  {var}=<valore>")
        print()
        print("Esempio di configurazione .env:")
        print("-" * 40)
        print("# Embedding Model Deployment")
        print("AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/")
        print("AZURE_OPENAI_EMBEDDING_KEY=your-embedding-api-key")
        print("AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002")
        print()
        print("# Chat Model Deployment")
        print("AZURE_OPENAI_CHAT_ENDPOINT=https://your-resource.openai.azure.com/")
        print("AZURE_OPENAI_CHAT_KEY=your-chat-api-key")
        print("AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4")
        print()
        print("AZURE_OPENAI_API_VERSION=2024-02-01")
        print("=" * 80)
        raise ValueError(f"Configurazione Azure mancante: {', '.join(missing)}")


# =========================
# Esecuzione dimostrativa
# =========================

def main():
    settings = SETTINGS
    
    # Validazione configurazione Azure
    validate_azure_configuration(settings)
    
    print("=" * 80)
    print("RAG Pipeline con Azure AI Foundry")
    print("-" * 80)
    print(f"Embedding Model: {settings.azure_embedding_deployment}")
    print(f"Chat Model: {settings.azure_chat_deployment}")
    print(f"Persist Directory: {settings.persist_dir}")
    print("=" * 80)
    print()

    # 1) Componenti Azure
    print("Inizializzazione modelli Azure...")
    embeddings = get_azure_embeddings(settings)
    llm = get_azure_chat_model(settings)

    # 2) Dati simulati e indicizzazione (load or build)
    print("Preparazione documenti...")
    docs = simulate_corpus()
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    print("Configurazione retriever...")
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    print("Costruzione catena RAG...")
    chain = build_rag_chain(llm, retriever)
    print()

    # 5) Esempi di domande
    questions = [
        "Che cos'è una pipeline RAG e quali sono le sue fasi principali?",
        "A cosa serve FAISS e quali capacità offre?",
        "Cos'è MMR e perché è utile durante il retrieval?",
        "Quale dimensione hanno gli embedding prodotti da all-MiniLM-L6-v2?",
        "Cos'è Azure AI Foundry e quali vantaggi offre?"
    ]

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        try:
            ans = rag_answer(q, chain)
            print(ans)
        except Exception as e:
            print(f"Errore durante l'elaborazione: {e}")
        print()


if __name__ == "__main__":
    main()