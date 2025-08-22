import httpx
from openai import AzureOpenAI, BadRequestError
import streamlit as st

if (
    not st.session_state.get("openai_api_key")
    or not st.session_state.get("openai_endpoint")
    or not st.session_state.get("openai_model")
):
    st.error("Please go to the Login page and provide your Azure OpenAI credentials.")
    st.switch_page("login.py")
    # st.stop()


"""
# AcademyGPT
"""

AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

httpx_client = httpx.Client(http2=True, verify=False)

client = AzureOpenAI(
    api_key=st.session_state["openai_api_key"],
    azure_endpoint=st.session_state["openai_endpoint"],
    api_version=AZURE_OPENAI_API_VERSION,
    http_client=httpx_client,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Write your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        except BadRequestError as e:
            response = e
    st.session_state.messages.append({"role": "assistant", "content": response})
