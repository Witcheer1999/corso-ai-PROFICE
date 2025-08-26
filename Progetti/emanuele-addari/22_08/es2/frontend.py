import streamlit as st

st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🚀",
    layout="wide"
)
nome_utente = ""

if 'nome_utente' not in st.session_state:
    st.session_state.nome_utente = nome_utente


st.session_state.nome_utente = st.sidebar.text_input("Come ti chiami?")
st.sidebar.title(f"Ciao {st.session_state.nome_utente}!")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("+1"):
    st.session_state.counter += 1
    st.balloons()

if st.button("-1"):
    st.session_state.counter -= 1
    st.balloons()

st.write(st.session_state.counter)