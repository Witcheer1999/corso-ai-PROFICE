import streamlit as st
import requests
import time

# Configurazione pagina
st.set_page_config(page_title="Login - Chat App", layout="centered")

# Inizializzazione stato sessione (deve essere in tutti i file)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "endpoint" not in st.session_state:
    st.session_state.endpoint = ""
if "deployment_name" not in st.session_state:
    st.session_state.deployment_name = "gpt-35-turbo"
if "api_version" not in st.session_state:
    st.session_state.api_version = "2023-05-15"

# Funzione per testare la connessione API
def test_connection(endpoint, api_key, deployment_name, api_version):
    completions_endpoint = f"{endpoint}/openai/deployments/{deployment_name}/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    payload = {
        "prompt": "Test",
        "max_tokens": 5,
        "temperature": 0.7
    }
    try:
        response = requests.post(
            url=completions_endpoint,
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return True, None
        else:
            error_msg = f"Errore {response.status_code}: {response.text}"
            return False, error_msg
    except Exception as e:
        error_msg = f"Errore di connessione: {str(e)}"
        return False, error_msg

# Pagina principale (login)
st.title("Configurazione Azure OpenAI")

with st.form("login_form"):
    endpoint = st.text_input("Endpoint URL", value=st.session_state.endpoint, placeholder="https://tuoendpoint.openai.azure.com")
    api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
    deployment_name = st.text_input("Nome Deployment", value=st.session_state.deployment_name, placeholder="gpt-35-turbo")
    api_version = st.text_input("API Version", value=st.session_state.api_version, placeholder="2023-05-15")
    
    submitted = st.form_submit_button("Connetti")
    
    if submitted:
        if not endpoint or not api_key or not deployment_name or not api_version:
            st.error("Tutti i campi sono obbligatori!")
        else:
            with st.spinner("Verifica connessione..."):
                success, error_msg = test_connection(endpoint, api_key, deployment_name, api_version)
                if success:
                    # Salva le credenziali in session state
                    st.session_state.authenticated = True
                    st.session_state.api_key = api_key
                    st.session_state.endpoint = endpoint
                    st.session_state.deployment_name = deployment_name
                    st.session_state.api_version = api_version
                    st.success("Connessione riuscita!")
                    
                    # Reindirizza alla pagina di chat
                    st.switch_page("pages/1_Chat.py")
                else:
                    st.error(f"Impossibile connettersi ad Azure OpenAI: {error_msg}")

# Mostra un messaggio se l'utente è già autenticato
if st.session_state.authenticated:
    st.info("Sei già autenticato. Puoi accedere alla chat dalla barra laterale.")
