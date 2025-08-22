from openai import AzureOpenAI, APIConnectionError, RateLimitError, APIStatusError
import streamlit as st

st.title("ChatGPT-like clone")

client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-05-01-preview",
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = st.secrets["AZURE_OPENAI_DEPLOYMENT"]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})