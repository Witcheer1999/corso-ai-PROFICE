import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
import streamlit as st

@retry(
    retry=retry_if_exception_type(RateLimitError),  # ritenta solo se è un RateLimitError (429)
    wait=wait_exponential(min=2, max=30),         # intervallo esponenziale tra retry: 2s → 4s → 8s ... max 30s
    stop=stop_after_attempt(5))
def get_response(prompt="Scrivimi una piccola biografia su Lionel Messi."):
    response = client.chat.completions.create(
        model=st.session_state.azure['deployment'],
        messages=[
            {"role": "system", "content": "Sei un assistente in grado di rispondere a domande generiche"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("Mini chat")

# Controlla se l'utente ha fatto il setup, perchè potrebbe switchare, tramite sidebar 
# alla pagina del chatbot senza aver inserito le credenziali corrette.
if "azure" not in st.session_state:
    st.error("⚠️ Devi prima configurare le credenziali nella pagina Setup.")
    st.stop()

# Recupero i dati di azure salvati nel session_state
cfg = st.session_state.azure

# Crea client con i dati utente
client = AzureOpenAI(
    api_key=cfg["api_key"],
    api_version=cfg["api_version"],
    azure_endpoint=cfg["endpoint"]
)


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
    #print('Input inserito dall\'utente: ', user_text)
    st.session_state.messages.append({"role": "user", "content": user_text}) # aggiungo il prompt inserito dall'utente nella lista dei messaggi
    response_llm = get_response(prompt=user_text) # eseguiamo il prompt
    #print('risposta llm: ', response_llm)
    st.session_state.messages.append({"role": "assistant", "content": response_llm}) # aggiungo la risposta dell'LLM nella lista dei messaggi
    st.rerun()  # forza il refresh della GUI con i messaggi aggiornati