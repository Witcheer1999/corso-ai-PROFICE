import os

from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st

# Load environment variables
load_dotenv()

# Get values from .env
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# login page
def login():
    st.header("Login")
    endpoint = st.text_input("Endpoint")
    key = st.text_input("Key", type="password")
    if st.button("Login"):
        if endpoint == ENDPOINT and key == KEY:
            st.session_state['endpoint'] = endpoint
            st.session_state['key'] = key
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

def chat():
    # Create Azure OpenAI client
    client = AzureOpenAI(
        api_key=st.session_state['key'],
        api_version=API_VERSION,
        azure_endpoint=st.session_state['endpoint']
    )
    
    cols = st.columns([4,1])
    # Add logout button
    with cols[1]:
        if st.button("Logout", key="logout_btn", type="primary"):
            st.session_state['logged_in'] = False
            st.session_state['endpoint'] = ""
            st.session_state['key'] = ""
            st.session_state.messages = []
            st.rerun()

    
    
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

login_page = st.Page(login, title="Log in")
chat_page = st.Page(chat, title="Chat")

st.title("Azure OpenAI Chatbot")

if st.session_state.get('logged_in', False):
    pg = st.navigation([chat_page])
else:
    pg = st.navigation([login_page])
    
pg.run()