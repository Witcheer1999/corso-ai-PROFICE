import httpx
from openai import AzureOpenAI, BadRequestError
import streamlit as st

AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

httpx_client = httpx.Client(http2=True, verify=False)

"""
# Welcome to AcademyGPT

## User Input
"""

api_key = st.text_input("Azure OpenAI API Key", type="password")
endpoint = st.text_input("Azure OpenAI Endpoint URL")
model = st.text_input("Model Name", value="gpt-4.1-nano")


if api_key and endpoint and model:
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=AZURE_OPENAI_API_VERSION,
        http_client=httpx_client,
    )

    try:
        available_models = client.models.list()

        if model not in [m.id for m in available_models.data]:
            st.error(f"Model '{model}' is not available. Please check the model name.")

    except BadRequestError as e:
        st.error(f"{e}")

    st.success("The provided credentials work and are set correctly!")
    # create button to go to pages/chat.py
    if st.button("Go to Chat"):
        st.session_state["openai_api_key"] = api_key
        st.session_state["openai_endpoint"] = endpoint
        st.session_state["openai_model"] = model
        st.switch_page("pages/chat.py")

else:
    st.warning("Please provide all the inputs.")
