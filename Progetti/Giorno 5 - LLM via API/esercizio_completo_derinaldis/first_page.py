import streamlit as st
from llm import ask_openai, get_client, validate_key_endpoint

endpoint = st.text_input("Enter your Azure OpenAI endpoint")
key = st.text_input("Enter your Azure OpenAI key")

if st.button("Validate"):
    if validate_key_endpoint(key, endpoint):
        st.success("Key and endpoint are valid.")
        client = get_client(endpoint, key)
        # passo lo user alla chat
        st.session_state.client = client
        st.success("Client created successfully.")
        st.switch_page("pages/app.py")
    else:
        st.error("Invalid key or endpoint.")