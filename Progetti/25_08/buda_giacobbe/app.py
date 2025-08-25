from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List
import getpass

from dotenv import load_dotenv
import streamlit as st

import faiss

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.chat_models import init_chat_model

from openai import AzureOpenAI

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

# Load environment variables
load_dotenv()

# Get values from .env
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
st.session_state.search_method = None
st.session_state.k = None
st.session_state.fetch_k = None
st.session_state.lambda_param = None
st.session_state.uploaded_files = None

def selection():
    # sidebar
    st.sidebar.header("Parameters")
    #if "search_method" not in st.session_state:    
    st.session_state.search_method = st.sidebar.selectbox("Search method", ["Select option", "MMR", "Similarity"], index=0, placeholder="Select a search method")
    if st.session_state.search_method == "MMR":
        st.session_state.lambda_param = st.sidebar.slider("Lambda", 0.0, 1.0, 0.5, 0.1)
        st.session_state.fetch_k = int(st.sidebar.number_input(label="Fetch K"))
        st.session_state.k = int(st.sidebar.number_input(label="K"))
    elif st.session_state.search_method == "Similarity":
        st.session_state.k = int(st.sidebar.number_input(label="K"))
    else:
        st.sidebar.warning("Select a search method")
        
    st.sidebar.header("Upload Documents")
    st.session_state.uploaded_files = st.sidebar.file_uploader("Upload your document (txt, md)", type=["txt", "md"], key="file_uploader", accept_multiple_files=True)
    
    st.session_state.ready = st.sidebar.button("Submit")
                
        

# =========================
# Configurazione
# =========================

load_dotenv()

    

# =========================
# Componenti di base
# =========================

def get_embeddings():

    if not os.getenv("AZURE_OPENAI_KEY"):
        os.environ["AZURE_OPENAI_KEY"] = getpass.getpass(
            "Enter your AzureOpenAI API key: "
        )
    
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def get_llm_from_lmstudio(settings):
    """
    Inizializza un ChatModel puntando a LM Studio (OpenAI-compatible).
    Richiede:
      - OPENAI_BASE_URL (es. http://localhost:1234/v1)
      - OPENAI_API_KEY (placeholder qualsiasi, es. "not-needed")
      - LMSTUDIO_MODEL (nome del modello caricato in LM Studio)
    """
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    model_name = os.getenv(settings["lmstudio_model_env"])

    if not base_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY devono essere impostate per LM Studio."
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile {settings['lmstudio_model_env']} con il nome del modello caricato in LM Studio."
        )

    return init_chat_model(model_name, model_provider="azure_openai", api_key=api_key, api_version=api_version)

def load_file(file) -> List[Document]:
    documents = []
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")  # decode bytes to string

    print(f"Loaded {file.name} ({len(content)} characters), type: {type(content)}")

    documents = [
        Document(
            page_content=section,
            metadata={"source": file.name, "section": i}
        )
        for i, section in enumerate(content.split("---"), start=1)
        if section.strip()
    ]
    
    return documents

def load_documents(uploaded_files) -> List[Document]:
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".md"):
                documents.extend(load_file(uploaded_file))
            else:
                raise ValueError(f"Unsupported file format")
    return documents
    

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


def split_documents(docs: List[Document], settings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings, embeddings, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings["persist_dir"])
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings["persist_dir"],
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings["persist_dir"])


def make_retriever(vector_store: FAISS, settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings["search_type"] == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings["k"], "fetch_k": settings["fetch_k"], "lambda_mult": settings["mmr_lambda"]},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings["k"]},
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
         "1) Rispondi solo con informazioni contenute nel contesto, ignora ogni tua altra precedente conoscenza.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) La risposta è sempre presente nel contesto.")
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


# =========================
# Esecuzione dimostrativa
# =========================

def rag():
    settings = {
        "persist_dir": "faiss_index_example",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "search_type": st.session_state.search_method,
        "k": st.session_state.k,
        "fetch_k": st.session_state.fetch_k,
        "mmr_lambda": st.session_state.lambda_param,
        "lmstudio_model_env": "LMSTUDIO_MODEL"
    }
    
    # 1) Componenti
    embeddings = get_embeddings()
    llm = get_llm_from_lmstudio(settings)

    # 2) Docuementi e indicizzazione
    docs = load_documents(st.session_state.uploaded_files)
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    # 5) Chat interattiva
    return chain

def print_chat(chain):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        ans = rag_answer(prompt, chain)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(ans)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ans})
            

def main():
    selection()
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
        
    if st.session_state.ready:
        if (st.session_state.search_method == "MMR" or st.session_state.search_method == "Similarity") and st.session_state.uploaded_files is not None:
            st.session_state.chain = rag()
        else:
            st.sidebar.warning("Set all parameters or upload at least one file!")
    
    if st.session_state.chain is not None:
        print_chat(st.session_state.chain)
    
if __name__ == "__main__":
    main()