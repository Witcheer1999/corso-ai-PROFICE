import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, stop_after_delay


# Initialize AzureOpenAI Client
load_dotenv("C:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/.env")  #C:/Users/LH668YN/OneDrive - EY/Desktop/
subscription_key = os.getenv("SUBSCRIPTION_KEY") or ""
azure_endpoint = os.getenv("AZURE_ENDPOINT") or ""
api_version = os.getenv("API_VERSION") or ""
deployment_name = os.getenv("DEPLOYMENT_NAME") or ""

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=subscription_key,
)


st.title('🤖 Your gpt-4o-mini Assistant')

# Initialize chat history the first time
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history so far
with st.container():
  for m in st.session_state.messages:
    with st.chat_message(m["role"]):
      st.markdown(m["content"])

# Wait for user input
prompt = st.chat_input("Say something")
if prompt:
  message = {"role": "user", "content": prompt}

  # Display user input
  st.session_state.messages.append(message)
  with st.container():
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Generate and display assistant response
  with st.chat_message("assistant"):
    stream_placeholder = st.empty()
    partial = ""
    for chunk in client.chat.completions.create(
        model=deployment_name,
        messages=st.session_state.messages,
        max_tokens=300,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=True,
    ):
      # Each chunk is a ChatCompletionChunk; text arrives in choices[0].delta.content
      if chunk.choices:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            partial += delta.content  #type: ignore
            stream_placeholder.markdown(partial)

  #update the session_state.messages for future reruns
  message_response = {"role": "assistant", "content": partial}
  st.session_state.messages.append(message_response)