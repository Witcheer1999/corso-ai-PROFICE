import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import streamlit as st

#vecchio codice
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
endpoint = os.getenv('OPENAI_ENDPOINT')

deployment = "gpt-35-turbo-instruct"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)


def get_completion_stream(prompt):
    response = client.completions.create(
        prompt=prompt,
        max_tokens=2000,# evita che vada in errore per risposte troppo lunghe
        temperature=1.0,
        top_p=1.0,
        model=deployment,
        stream=True
    )
    full_text = ""
    '''prendi i chunks che arrivano da responce, verifica che choices esista e che contenga text'''
    for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "text"):
            full_text += chunk.choices[0].text
            yield full_text


st.title("ChatGPT Fasullo")

# inizializza la chat e l'input
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_value' not in st.session_state:
    st.session_state.input_value = ""


'''visualizza tutti i messaggi nella cronoogia (history) controlla se è scritto da user
o scritto dal chatbot, utilizza due stili di markdown per mostrarli'''
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

user_input = st.chat_input("Scrivi il tuo messaggio...")
'''controlla se user ha scritto qualcosa, crea il placeholder per la risposta
e crea anche la variabile per la rissposta completa, una volta ottenuti i chunk aggiorna il placeholder, alla fine
salva in cronologia, se ci sta l errore salva l errore in cronologia'''
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            for partial in get_completion_stream(user_input):
                full_response = partial
                response_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = f"Errore: {e}"
            response_placeholder.markdown(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})