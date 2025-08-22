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
    api_version = st.text_input("API version", value="2025-01-01-preview")
    subscription_key = st.text_input("Subscription key", value="988oydbOEbkuVr92EZf2CpGKHs5FuPoNksa7p6eyB7M8oKuumMOcJQQJ99BHACYeBjFXJ3w3AAAAACOGYDOO")
    endpoint = st.text_input("Endpoint", value="https://instance-foundry-0.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview")

    # Bottone per inviare il form
    submitted = st.form_submit_button("Connect")

    if submitted:

        client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )
        
        try:

            client.models.list()
            
            if deployment_name not in [m.id for m in available_models.data]:
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
