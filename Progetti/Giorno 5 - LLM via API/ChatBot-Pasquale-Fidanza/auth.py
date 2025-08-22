import streamlit as st
from openai import AzureOpenAI  # ✅ Assicurati di avere openai>=1.0.0 con supporto Azure

st.title("Azure Auth")

# Inizializza session state per le credenziali
if "azure_session" not in st.session_state:
    st.session_state.azure_session = None

with st.form("login_form"):
    endpoint = st.text_input("Azure Endpoint", placeholder="https://<nome>.openai.azure.com/")
    key = st.text_input("Azure API Key", type="password")
    deployment = 'gpt-35-turbo'
    api_version = '2024-12-01-preview'
    submit = st.form_submit_button("Autenticati")

    if submit:
        if endpoint and key:

            try:
                # Se la sessione viene inizializzata, passa alla chat
                azure_client = AzureOpenAI(
                    api_key=key,
                    azure_endpoint=endpoint,
                    api_version=api_version
                )

                st.session_state.azure_session = {
                    "endpoint": endpoint,
                    "key": key,
                    "deployment": deployment,
                    "api_version": api_version
                }
                st.success("Autenticazione completata! Vai alla pagina Chat.")
                st.switch_page('pages/llm.py')
            except Exception as e:
                st.error(f"Errore di connessione: {e}")
        else:
            st.error("Compila tutti i campi.")
