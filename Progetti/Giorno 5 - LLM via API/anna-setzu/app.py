import streamlit as st

st.title('LLM chat')

#definisco le pagine
login_page = st.Page('login.py', title='Login')
chat_page = st.Page('chat.py', title='Chat')

#navigazione
pg = st.navigation([login_page, chat_page])
pg.run()

