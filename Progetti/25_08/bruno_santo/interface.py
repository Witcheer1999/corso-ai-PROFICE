import streamlit as st
from io import StringIO
from es1.embedding import define_llm,define_embedder,rag_answer,build_rag_chain, define_vector_db,make_retriever,split_documents,Settings
from langchain.schema import Document


st.set_page_config(page_title="Azure OpenAI Chat", layout="wide")

settings = Settings()
llm = define_llm(settings)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    files = st.file_uploader("Caricare file", [".pdf", ".txt"],accept_multiple_files=True)
    if len(files) != 0:
        docs = []
        #dobbiamo splittare in chunk ogni file in input 
        for file in files:
            stringio = StringIO(file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            docs.append(Document(page_content = string_data))

        chunks = split_documents(docs)
        client_embeddings = define_embedder(settings)
        vector_store = define_vector_db(settings, chunks,client_embeddings)
        retriever = make_retriever(vector_store, settings)
        chain = build_rag_chain(llm, retriever)


user_response = st.chat_input(placeholder="Inserisci un messaggio ")
messages = st.container()
if user_response:
    #risposta del chatbot
    
    messages.chat_message("user").write(user_response)
    bot_response = rag_answer(user_response, chain)
    messages.chat_message("assistant").write(f"{bot_response}")
