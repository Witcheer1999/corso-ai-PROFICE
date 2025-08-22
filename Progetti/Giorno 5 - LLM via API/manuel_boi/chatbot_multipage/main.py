import streamlit as st
from openai import AzureOpenAI


st.set_page_config(page_title="Connessione Azure", layout="centered")

def test_connection(endpoint, deployment, api_key, api_version):
    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        client.chat.completions.create(
            messages=[{"role": "system", "content": "Ping"}],
            model=deployment,
            temperature=0.0,
            max_tokens=10
        )
        return client
    except Exception as e:
        return str(e)

st.title("Connetti a Azure OpenAI")

with st.form("connessione_form"):
    endpoint = st.text_input("Endpoint Azure OpenAI")
    deployment = st.text_input("Nome Deployment")
    api_key = st.text_input("Chiave API", type="password")
    api_version = st.text_input("Versione API (es. 2023-07-01)")
    submitted = st.form_submit_button("Connetti")

    if submitted:
        result = test_connection(endpoint, deployment, api_key, api_version)
        if isinstance(result, AzureOpenAI):
            st.session_state.client = result
            st.session_state.endpoint = endpoint
            st.session_state.deployment = deployment
            st.session_state.api_version = api_version
            st.session_state.api_key = api_key
            st.session_state.messages = []
            st.switch_page("pages/chat.py")
        else:
            st.error(f"Connessione fallita: {result}")
