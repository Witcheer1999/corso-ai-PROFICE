import streamlit as st
from utils import test_connection, deployment

st.set_page_config(page_title="Login", page_icon="🔧")

st.title("🔧 Configurazione Azure OpenAI")

with st.form("config"):
    endpoint = st.text_input("Endpoint", placeholder="https://your-resource.openai.azure.com/")
    api_key = st.text_input("API Key", type="password")

    if st.form_submit_button("Connetti"):
        if not all([endpoint, api_key]):
            st.error("Tutti i campi sono obbligatori!")
        else:
            success, result = test_connection(endpoint, api_key)
            if success:
                st.session_state.client = result
                st.session_state.deployment = deployment
                st.success("Connesso con successo! Ora vai alla pagina Chat.")
            else:
                st.error(f"Errore: {result}")
