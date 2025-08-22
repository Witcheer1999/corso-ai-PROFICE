import streamlit as st
from openai import AzureOpenAI


subscription_key = st.session_state["subscription_key"]
azure_endpoint = st.session_state["azure_endpoint"]
api_version = st.session_state["api_version"]
deployment_name = st.session_state["deployment_name"]


if not st.session_state.get("auth_ok"):
    st.warning("Configura le credenziali nella pagina di Setup.")
    st.page_link("app.py", label="← Torna al Setup")
    st.stop()

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
        max_tokens=500,
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


""" Equivalent ways to run code with streamlit:

python -m streamlit run 'C:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/class_excercises/08_22_class-5/02_streamlit/streamlit_LLMchat.py'
streamlit run "C:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/class_excercises/08_22_class-5/02_streamlit/streamlit_LLMchat.py" 

"""