import streamlit as st
from auth_utils import init_session_state, ensure_login, call_model, logout, mask_key

init_session_state()
ensure_login()

st.set_page_config(page_title="Chat Azure OpenAI", page_icon="💬")

# Sidebar
with st.sidebar:
    st.subheader("Sessione")
    st.caption(f"Endpoint: {st.session_state.endpoint}")
    st.caption(f"Deployment: {st.session_state.deployment}")
    st.caption(f"API Key: {mask_key(st.session_state.api_key)}")
    if st.button("Logout"):
        logout()
    st.divider()
    st.text_area("System prompt", key="system_prompt", height=100)
    if st.button("Reset chat"):
        st.session_state.messages = []
        st.experimental_rerun()

st.title("💬 Chatbot Azure OpenAI")

# Inizializza messaggi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra cronologia
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input chat
prompt = st.chat_input("Scrivi un messaggio...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            assistant_reply = call_model(prompt, [m for m in st.session_state.messages if m["role"] != "system" and m["role"] != "assistant" or True][:-1])
            # Nota: sopra usiamo un filtraggio minimale; meglio passare tutta la cronologia user/assistant
            # Rifacciamo correttamente:
            history = [m for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")]
            assistant_reply = call_model(prompt, history)
            st.markdown(assistant_reply)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        except Exception as e:
            st.error(f"Errore: {e}")
