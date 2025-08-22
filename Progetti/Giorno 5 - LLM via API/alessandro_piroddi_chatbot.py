from openai import AzureOpenAI
import streamlit as st
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni le configurazioni dall'ambiente
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

# Verifica che tutte le variabili d'ambiente siano impostate
if not all([api_version, azure_endpoint, api_key, deployment]):
    st.error("⚠️ Le variabili d'ambiente per Azure OpenAI non sono configurate correttamente.")
    st.info("Assicurati di aver impostato AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY e AZURE_OPENAI_DEPLOYMENT nel file .env")
    st.stop()


client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Configura la pagina Streamlit
st.set_page_config(
    page_title="Azure OpenAI Chatbot",
    layout="wide"
)

st.title("Chatbot con Azure OpenAI")
st.write(f"Questo chatbot utilizza il deployment Azure OpenAI: **{deployment}**")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = deployment  # Usa il deployment di Azure anziché il nome del modello

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar per opzioni e informazioni
with st.sidebar:
    st.header("Informazioni")
    st.write(f"**API Version**: {api_version}")
    st.write(f"**Deployment**: {deployment}")
    
    st.header("Opzioni")
    if st.button("Cancella chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Chatbot creato da Alessandro Piroddi")

# Visualizza i messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})