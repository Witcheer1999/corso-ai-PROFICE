import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Carica il file .env
load_dotenv()

# Recupera le variabili dal .env
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Inizializza il client
@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

client = get_openai_client()

def ask_openai(user_prompt):
    return client.completions.create(
        model=deployment,
        prompt=user_prompt,
        max_tokens=500,
        temperature=1.0
    )

# Interfaccia Streamlit
st.title("Chatbot con Azure OpenAI")

# Inizializza la cronologia della chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra la cronologia dei messaggi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
prompt = st.chat_input("")

if prompt:
    # Aggiungi il messaggio dell'utente alla cronologia
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostra il messaggio dell'utente
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Ottieni la risposta da Azure OpenAI
    with st.chat_message("assistant"):
            try:
                response = ask_openai(prompt)
                assistant_response = response.choices[0].text.strip()
                st.markdown(assistant_response)
                
                # Aggiungi la risposta dell'assistente alla cronologia
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                st.error(f"Errore nella chiamata ad Azure OpenAI: {str(e)}")