import os
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from typing import List
import tempfile
 
import faiss
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
# ==================================
# Config e settings
# ==================================
load_dotenv()
 
AZURE_OPENAI_KEY = os.getenv("KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("DEPLOYMENT")
AZURE_API_VERSION = os.getenv("VERSION")
LLM_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT")
 
@dataclass
class Settings:
    chunk_size: int = 700
    chunk_overlap: int = 100
    search_type: str = "similarity"
    k: int = 5
    fetch_k: int = 10
    mmr_lambda: float = 0.3
 
SETTINGS = Settings()
 
# ==================================
# Funzioni core
# ==================================
def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        model=EMBEDDING_DEPLOYMENT
    )
 
def get_llm(settings: Settings):
    return init_chat_model(
        LLM_DEPLOYMENT,
        model_provider="azure_openai",
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
 
def load_documents_from_uploaded_files(uploaded_files) -> List[Document]:
    """
    Converte i file caricati via Streamlit (txt, md, pdf) in Document LangChain.
    """
    documents = []
    for uploaded_file in uploaded_files:
        file_suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
 
        if file_suffix == ".pdf":
            loader = UnstructuredPDFLoader(tmp_file_path)
        elif file_suffix in [".txt", ".md"]:
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        else:
            continue  
 
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            documents.append(doc)
 
    return documents
 
def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n---\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)
 
def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings) -> FAISS:
    return FAISS.from_documents(documents=chunks, embedding=embeddings)
 
def make_retriever(vector_store: FAISS, settings: Settings):
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": settings.k})
 
def format_docs_for_prompt(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)
 
def build_rag_chain(llm, retriever):
    system_prompt = (
        "Sei un assistente che risponde **solo ed esclusivamente** con le informazioni "
        "contenute nel contesto fornito, anche se sono errate. "
        "Non devi correggere, modificare o filtrare nulla: usa le risposte così come sono scritte. "
        "Il tuo compito è quello di restituire esattamente le risposte del documento, "
        "anche se sono palesemente sbagliate.\n\n"
        "Includi sempre una citazione tra parentesi quadre nel formato [source:...]. "
        "Rispondi in italiano, in modo conciso."
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
 
# ==================================
# Streamlit App
# ==================================
def main():
    st.set_page_config(page_title="ChatBot RAG", page_icon="🤖")
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs" not in st.session_state:
        st.session_state.docs = []
 
    st.title("ChatBot RAG")
 
    # Sidebar configuration
    st.sidebar.header("Parametri")
 
    SETTINGS.search_type = st.sidebar.selectbox("Search type:", ["similarity", "mmr"], index=0)
    SETTINGS.k = st.sidebar.number_input("k (top risultati):", min_value=1, max_value=10, value=5)
 
    uploaded_files = st.sidebar.file_uploader(
        "Carica i file (txt, PDF, md):",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True
    )
 
    if st.sidebar.button("Carica i documenti"):
        if uploaded_files:
            docs = load_documents_from_uploaded_files(uploaded_files)
            st.session_state.docs = docs
            st.sidebar.success(f"{len(docs)} documenti caricati con successo!")
        else:
            st.sidebar.warning("Nessun file selezionato.")
 
    # Costruzione componenti se ci sono documenti
    if st.session_state.docs:
        @st.cache_resource
        def init_components(_docs, settings):
            embeddings = get_embeddings(settings)
            llm = get_llm(settings)
            chunks = split_documents(_docs, settings)
            vector_store = build_faiss_vectorstore(chunks, embeddings)
            retriever = make_retriever(vector_store, settings)
            chain = build_rag_chain(llm, retriever)
            return chain
 
        chain = init_components(st.session_state.docs, SETTINGS)
 
 
        # Mostra conversazione
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
 
        # Input utente
        if prompt := st.chat_input("Fai una domanda sui documenti..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
 
            with st.chat_message("assistant"):
                with st.spinner("Sto pensando..."):
                    response = chain.invoke(prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("📂 Carica dei documenti dalla sidebar per iniziare.")
 
if __name__ == "__main__":
    main()
 
 