import streamlit as st
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Get values from .env
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT
)

# Streamlit app

st.title("My first chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=st.session_state.messages,
        max_tokens=300,
        stream=True
)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})