import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Carica variabili d'ambiente
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Inizializza il client Azure OpenAI
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

st.title("Chatbot con GPT-4o-mini")

# Inizializza cronologia messaggi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utente
if prompt := st.chat_input("Scrivi qui..."):
    # Salva messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Streaming reale della risposta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=deployment,
            temperature=1.0,
            max_tokens=4096,
            stream=True
        )

        for chunk in stream:
            if (
                chunk.choices and
                chunk.choices[0].delta and
                hasattr(chunk.choices[0].delta, "content") and
                chunk.choices[0].delta.content
            ):
                streamed_text += chunk.choices[0].delta.content
                placeholder.markdown(streamed_text)

    # Salva risposta nella cronologia
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})
