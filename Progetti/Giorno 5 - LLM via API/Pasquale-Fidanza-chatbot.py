import os
import streamlit as st
import random, time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Configurazione Azure OpenAI
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

# Inizializza la cronologia messaggi
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Funzione per inviare messaggi ad Azure OpenAI
def send_input():
    user_msg = st.session_state.user_input
    if not user_msg:
        return

    # Salva il messaggio dell'utente
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Chiamata API con tutta la cronologia
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are a helpful assistant. Answer in Italian."}] 
                 + st.session_state.messages,
        max_tokens=1000,
        temperature=1.0,
        top_p=1.0,
        model=AZURE_OPENAI_DEPLOYMENT
    )

    # Prendi il contenuto della risposta dell’assistente
    assistant_msg = response.choices[0].message["content"]

    # Salva la risposta nella cronologia
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Input chat
st.chat_input("Inserisci il tuo prompt", key="user_input", on_submit=send_input)

# Mostra l’ultima risposta dell’assistente
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
