import os

from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI

st.title("ChatGPT-like clone")

# Carica le variabili d'ambiente dal file .env
# load_dotenv(dotenv_path=".env")

# Recupera le credenziali
api_key = st.session_state["AZURE_API_KEY"]
endpoint = st.session_state["AZURE_ENDPOINT"]
deployment = st.session_state["AZURE_DEPLOYMENT"]

client = AzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            model=deployment
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})