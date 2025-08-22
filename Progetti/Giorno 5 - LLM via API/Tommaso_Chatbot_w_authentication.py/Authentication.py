from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv("Lezione_5/.env")

# Load environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
connection_string = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def Authentication(key, endpoint):
    key_incorrect = key != api_key
    endpoint_incorrect = endpoint != connection_string

    if not key_incorrect and not endpoint_incorrect:
        return "Autenticato"
    else:
        return "ERRORE AUTENTICAZIONE! Credenziali errate."

st.title("Login Azure OpenAI")

key_input = st.text_input("Inserisci la tua API Key", type="password")
endpoint_input = st.text_input("Inserisci l'Endpoint Azure")

if st.button("Login"):
    result = Authentication(key_input, endpoint_input)
    if result == "Autenticato":
        st.success(result)
        st.session_state.authenticated = True
        st.session_state.api_key = key_input
        st.session_state.endpoint = endpoint_input
        st.session_state.deployment = deployment
        st.switch_page("pages/Chatbot.py")  
    else:
        st.error(result)
