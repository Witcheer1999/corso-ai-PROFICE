import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError


# Carica variabili dal file .env
load_dotenv(dotenv_path="")

# Recupero variabili ambiente
openai_api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Creo client Azure OpenAI
client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

@retry(
    retry=retry_if_exception_type(RateLimitError),  # ritenta solo se è un RateLimitError (429)
    wait=wait_exponential(min=2, max=30),         # intervallo esponenziale tra retry: 2s → 4s → 8s ... max 30s
    stop=stop_after_attempt(5))
def get_response(prompt="Scrivimi una piccola biografia su Lionel Messi."):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Sei un assistente in grado di rispondere a domande generiche"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.2
    )
    return response.choices[0].message.content


import streamlit as st

st.title("Mini chat")

# Stato: messages -> lista contenente storico messaggi (role: 'user' o 'assistant')
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Come posso esserti utile?"}]

# Render: assistant a sinistra, user a destra
for mex in st.session_state.messages:
    left, right = st.columns(2)
    if mex["role"] == "assistant":
        with left:
            st.write(f"🤖 {mex['content']}")
    else:
        with right:
            st.write(f"🧑 {mex['content']}")


# Input in basso (si svuota da solo dopo l'invio)
user_text = st.chat_input("Scrivi un messaggio...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text}) # aggiungo il prompt inserito dall'utente nella lista dei messaggi
    response_llm = get_response(prompt=user_text) # eseguiamo il prompt
    st.session_state.messages.append({"role": "assistant", "content": response_llm}) # aggiungo la risposta dell'LLM nella lista dei messaggi
    st.rerun()                                                                      # forza il refresh della GUI con i messaggi aggiornati
