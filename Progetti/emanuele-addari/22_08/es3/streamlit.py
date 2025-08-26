import streamlit as st
from openai_client import Ciatgpt

st.set_page_config(
    page_title="obenai",
    page_icon="🚀",
    layout="wide"
)

if "api_key" not in st.session_state or "conn_str" not in st.session_state:
    st.title("Configurazione OpenAI")
    api_key = st.text_input("API Key:", type="password")
    conn_str = st.text_input("Endpoint:", type="password")
    if st.button("Prosegui"):
        if not api_key or not conn_str:
            st.error("Entrambi i campi sono obbligatori.")
            st.stop()
        st.session_state.api_key = api_key
        st.session_state.conn_str = conn_str
    st.stop()

if "ciatgpt" not in st.session_state:
    st.session_state.ciatgpt = Ciatgpt(api_key=st.session_state.api_key, conn_str=st.session_state.conn_str)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

user_input = st.text_input("Sono il magico chatgpt fammi una domanda:")

if user_input:
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    streamed_text = ""
    for chunk in st.session_state.ciatgpt.answer(user_input):
        streamed_text += chunk
    st.session_state.conversation_history.append({"role": "assistant", "content": streamed_text})

for i in range(0, len(st.session_state.conversation_history), 2):
    user = st.session_state.conversation_history[i]
    response = st.session_state.conversation_history[i + 1] if i + 1 < len(st.session_state.conversation_history) else None
    st.write(f"**User:** {user['content']}")
    st.write(f"**Magic ChatGPT:** {response['content'] if response else 'No response'}")