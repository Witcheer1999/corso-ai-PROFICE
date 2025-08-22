# Home page streamlit with a form including AZURE_ENDPOINT and AZURE_API_KEY
from openai import AzureOpenAI, api_key
import streamlit as st
st.title("Home Page")

def change_page():
    # Go to schermata_chatgpt.py
    st.switch_page("pages/schermata_chatgpt.py")

with st.form("azure_form"):
    st.session_state["AZURE_DEPLOYMENT"] = st.text_input("AZURE_DEPLOYMENT")
    st.session_state["AZURE_ENDPOINT"] = st.text_input("AZURE_ENDPOINT")
    st.session_state["AZURE_API_KEY"] = st.text_input("AZURE_API_KEY", type="password")
    submit = st.form_submit_button("Submit")

# When the submit is pressed
    if submit:

        # Check if the info is valid
        client = AzureOpenAI(
            api_version="2025-01-01-preview",
            azure_endpoint=st.session_state["AZURE_ENDPOINT"],
            api_key=st.session_state["AZURE_API_KEY"],
        )

        # Check if the connection is valid
        try:
            client.chat.completions.create(
                model=st.session_state["AZURE_DEPLOYMENT"],
                messages=[
                    {"role": "user", "content": "Hello, world!"}
                ]
            )
            st.success("Connection is valid!")

            change_page()

        except Exception as e:
            st.error(f"Connection failed: {e}")

    