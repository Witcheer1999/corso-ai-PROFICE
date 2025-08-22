import streamlit as st
from openai import AzureOpenAI



def connetti():
    st.title("Inserisci dati per accedere al modello Azure OpenAI")

    # Inizializza variabili globali
    if "client" not in st.session_state:
        st.session_state.client = None
    if "deployment" not in st.session_state:
        st.session_state.deployment = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    endpoint = st.text_input("Endpoint Azure:", placeholder="https://xxxx.openai.azure.com/")
    api_key = st.text_input("API Key:", type="password")
    api_version = st.text_input("API Version:", placeholder="2024-05-01-preview")
    deployment = st.text_input("Deployment Name:", placeholder="gpt-35-turbo")


    if st.button("Connetti"):
        if endpoint and api_key and api_version and deployment:
            #prova()
            try:
                client = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                )
                _ = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                st.session_state.client = client
                st.session_state.deployment = deployment
                st.success("Connessione riuscita! Attendi qualche istante e verrai reindirizzato alla chat.")
                #st.session_state["logged_in"] = True
                #st.session_state.login_container.empty()
                change_page()
            except Exception as e:
                st.error(f"Errore di connessione: {e}\nRiprova ad inserire le credenziali correttamente.")
            
        else:
            st.warning("Compila tutti i campi prima di continuare.")


def change_page():
    # Go to pages/pages2.py
    st.switch_page("pages/page2.py")

connetti()
