import streamlit as st
from openai import AzureOpenAI

st.title("ChatBot")

# Controlla se l'utente ha effettuato il login
if not st.session_state.get("azure_session"):
    st.warning("Devi prima autenticarti nella pagina Login.")
    st.stop()

# Recupera le credenziali dalla session_state
azure_session = st.session_state.azure_session

client = AzureOpenAI(
    api_key=azure_session["key"],
    azure_endpoint=azure_session["endpoint"],
    api_version=azure_session["api_version"]
)
deployment = azure_session["deployment"]

# Inizializza la cronologia messaggi
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    user_msg = st.session_state.user_input
    if not user_msg:
        return

    st.session_state.messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": "You are a helpful assistant. Answer in Italian."}] + st.session_state.messages,
        max_tokens=1000,
        temperature=1.0,
        top_p=1.0
    )

    assistant_msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

st.chat_input("Inserisci il tuo prompt", key="user_input", on_submit=send_input)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
