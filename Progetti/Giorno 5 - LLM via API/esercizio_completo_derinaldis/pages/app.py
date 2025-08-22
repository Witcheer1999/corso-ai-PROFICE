import streamlit as st
from llm import ask_openai

client = st.session_state.client

st.title("Chat")

# #Domanda-risposta con l'LLM
# if st.button("Send"):
#     response = ask_openai(user_input)
#     st.text_area("LLM:", value=response, height=300)

# Inizializzo la chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utente stile chat
if prompt := st.chat_input("Scrivi un messaggio..."):
    # Mostra subito il messaggio dell’utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ask_openai(prompt, client)

    # Mostra la risposta e la salva
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)