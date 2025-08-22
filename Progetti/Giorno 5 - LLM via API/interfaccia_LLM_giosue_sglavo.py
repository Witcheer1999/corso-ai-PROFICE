import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

st.title("Chat Bot")

# Carica le variabili dal file .env
load_dotenv()

# Recupera le variabili
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Inizializza il client AzureOpenAI
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
# Inizializza lo stato della sessione
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-35-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi della chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Inserisco la barra del prompt
prompt = st.chat_input("Ask anything")

# Gestisco l'input dell'utente
if prompt:

    # Aggiungi il messaggio dell'utente allo stato della sessione
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostra il messaggio dell'utente
    with st.chat_message("user"):
        st.markdown(prompt)

    # Risposta del modello
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=200,
            temperature=1.0,
            top_p=1.0,
            stream = True
        )
        response = st.write_stream(stream)

    # Aggiungi la risposta del modello allo stato della sessione
    st.session_state.messages.append({"role": "assistant", "content": response})
