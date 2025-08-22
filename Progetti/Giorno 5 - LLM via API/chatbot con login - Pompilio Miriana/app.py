import streamlit as st
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

st.set_page_config(page_title="Azure OpenAI Chat", page_icon="💬")
st.title("Azure OpenAI Chatbot")

# Le pagine saranno gestite da Streamlit automaticamente se usiamo la cartella `pages/`
st.write("Usa il menu a sinistra per navigare tra le pagine.")
