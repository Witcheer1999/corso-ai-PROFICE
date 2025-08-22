import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Carica eventuali default da .env (opzionali, verranno sovrascritti dal login)
load_dotenv()

DEFAULT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
DEFAULT_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or ""
)
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# --- Gestione autenticazione semplice (solo in memoria di sessione) ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "endpoint" not in st.session_state:
    st.session_state.endpoint = DEFAULT_ENDPOINT
if "api_key" not in st.session_state:
    st.session_state.api_key = DEFAULT_API_KEY
if "deployment" not in st.session_state:
    st.session_state.deployment = DEFAULT_DEPLOYMENT

@st.cache_resource(show_spinner=False)
def _build_client(endpoint: str, api_key: str):
    if not (endpoint and api_key):
        return None
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

def get_client():
    if not st.session_state.authenticated:
        return None
    return _build_client(st.session_state.endpoint, st.session_state.api_key)

def logout():
    for k in ["authenticated", "endpoint", "api_key", "deployment", "messages"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Sidebar: stato e logout
with st.sidebar:
    st.markdown("### Credenziali")
    if st.session_state.authenticated:
        masked = st.session_state.api_key[:4] + "..." + st.session_state.api_key[-4:] if len(st.session_state.api_key) > 8 else "***"
        st.success("Connesso")
        st.caption(f"Endpoint: {st.session_state.endpoint}")
        st.caption(f"API Key: {masked}")
        if st.button("Logout"):
            logout()
    else:
        st.info("Non autenticato")

# Schermata di login (prima della chat)
if not st.session_state.authenticated:
    st.title("Login Azure OpenAI")
    st.write("Inserisci le credenziali (non vengono salvate su disco).")
    with st.form("login_form"):
        endpoint_in = st.text_input("Endpoint Azure", value=st.session_state.endpoint, placeholder="https://<nome-risorsa>.openai.azure.com")
        deployment_in = st.text_input("Deployment / Model name", value=st.session_state.deployment, placeholder="es. gpt-4o-mini")
        api_key_in = st.text_input("API Key", value=st.session_state.api_key, type="password")
        submitted = st.form_submit_button("Connetti")
    if submitted:
        if not (endpoint_in and api_key_in and deployment_in):
            st.error("Compila tutti i campi.")
        else:
            st.session_state.endpoint = endpoint_in.strip()
            st.session_state.api_key = api_key_in.strip()
            st.session_state.deployment = deployment_in.strip()
            # Test veloce creazione client
            try:
                tmp_client = _build_client(st.session_state.endpoint, st.session_state.api_key)
                # Chiamata minima: non sempre è disponibile un list models, quindi ci limitiamo a controllare l'oggetto
                if tmp_client is None:
                    st.error("Impossibile creare il client. Verifica i dati.")
                else:
                    st.session_state.authenticated = True
                    st.success("Connessione riuscita!")
                    st.rerun()
            except Exception as e:
                st.error(f"Errore connessione: {e}")
    st.stop()

client = get_client()
deployment = st.session_state.deployment

def ask_openai(user_prompt: str):
    if client is None:
        raise RuntimeError("Client non inizializzato: effettua il login")
    # Heuristica semplice per distinguere API chat
    if deployment and any(key in deployment.lower() for key in ["gpt", "turbo", "o", "mini"]):
        return client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=500,
            temperature=1.0,
        )
    # Fallback completions legacy
    return client.completions.create(
        model=deployment,
        prompt=user_prompt,
        max_tokens=500,
        temperature=1.0,
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
                # Determina se risposta è chat o completion
                if hasattr(response.choices[0], "message"):
                    assistant_response = response.choices[0].message.content.strip()
                else:
                    assistant_response = response.choices[0].text.strip()
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"Errore nella chiamata ad Azure OpenAI: {str(e)}")
