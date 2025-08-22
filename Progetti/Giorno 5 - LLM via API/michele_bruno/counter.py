import streamlit as st
from es1.script import ask
# Set up the page configuration
st.set_page_config(page_title="Chat App")

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Simple Chat")

user_input = st.text_input("Inserisci un messaggio:")

if st.button("Send"):
    #salva il messaggio dell'utente
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        response = ask(user_input)
        assistant_response = response.choices[0].message.content

        #mostro a video la risposta
        st.write(f"{assistant_response}")

        #salvo il messaggio del modello
        st.session_state.messages.append({"role":"assistant", "content":assistant_response})
    except Exception as e :
        st.error(f"{e}")



if st.button("Show History Chat"):
    for messages in st.session_state.messages:
        if messages["role"] == "user":
            st.write(messages["content"])