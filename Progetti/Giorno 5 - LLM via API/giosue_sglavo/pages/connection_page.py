import streamlit as st
from openai import AzureOpenAI
from openai import OpenAIError, BadRequestError

st.title("Azure OpenAI Connection")

deployment_name = "gpt-35-turbo"

# Inizializza lo stato della sessione
if "connection" not in st.session_state:
    st.session_state["connection"] = False

if "client" not in st.session_state:
    st.session_state["client"] = None

# Creazione del form
with st.form("connection_form"):
    api_version = st.text_input("API version")
    subscription_key = st.text_input("Subscription key")
    endpoint = st.text_input("Endpoint")

    # Bottone per inviare il form
    submitted = st.form_submit_button("Connect")

    if submitted:

        client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )
        
        try:

            returned_models = client.models.list()
            
            if deployment_name not in [m.id for m in returned_models.data]:
                st.error(f"Model '{deployment_name}' is not available. Please check the model name.")

        except BadRequestError as e:
            st.session_state["connection"] = False
            st.error(f"{e}")

        st.success("Connessione avvenuta con successo!")

        #Setto la connessione a True
        st.session_state["connection"] = True
        st.session_state["client"] = client

        print(st.session_state["connection"])

        st.switch_page("pages/llm_page.py")
