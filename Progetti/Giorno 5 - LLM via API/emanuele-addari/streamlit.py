import streamlit as st
from openai_client import Ciatgpt

st.set_page_config(
    page_title="obenai",
    page_icon="🚀",
    layout="wide"
)

if "ciatgpt" not in st.session_state:
    st.session_state.ciatgpt = Ciatgpt()
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