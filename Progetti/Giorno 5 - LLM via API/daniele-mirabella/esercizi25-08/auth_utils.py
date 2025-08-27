import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
DEFAULT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
DEFAULT_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or ""
)

SESSION_KEYS = ["authenticated", "endpoint", "api_key", "deployment", "messages", "system_prompt"]

def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    st.session_state.setdefault("endpoint", DEFAULT_ENDPOINT)
    st.session_state.setdefault("api_key", DEFAULT_API_KEY)
    st.session_state.setdefault("deployment", DEFAULT_DEPLOYMENT)
    st.session_state.setdefault("system_prompt", "Sei un assistente utile e conciso.")

@st.cache_resource(show_spinner=False)
def build_client(endpoint: str, api_key: str):
    if not (endpoint and api_key):
        return None
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

def get_client():
    if not st.session_state.get("authenticated"):
        return None
    return build_client(st.session_state.endpoint, st.session_state.api_key)

def logout():
    for k in SESSION_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def mask_key(key: str) -> str:
    if not key:
        return "***"
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "***"

def ensure_login():
    """Call at top of a page that requires auth. If not logged in, show info and stop."""
    if not st.session_state.get("authenticated"):
        st.warning("Devi prima effettuare il login dalla pagina Login.")
        st.stop()

def call_model(prompt: str, messages_history: list):
    client = get_client()
    if client is None:
        raise RuntimeError("Client non inizializzato")
    deployment = st.session_state.deployment
    # Costruisci lista messaggi per chat API
    chat_messages = []
    system_prompt = st.session_state.get("system_prompt")
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    chat_messages.extend(messages_history)
    chat_messages.append({"role": "user", "content": prompt})

    # Heuristica per decidere endpoint chat
    if deployment and any(k in deployment.lower() for k in ["gpt", "turbo", "o", "mini"]):
        response = client.chat.completions.create(
            model=deployment,
            messages=chat_messages,
            max_tokens=500,
            temperature=1.0,
        )
        if hasattr(response.choices[0], "message"):
            return response.choices[0].message.content.strip()
        return response.choices[0].text.strip()

    # Fallback completions
    response = client.completions.create(
        model=deployment,
        prompt=prompt,
        max_tokens=500,
        temperature=1.0,
    )
    return response.choices[0].text.strip()
