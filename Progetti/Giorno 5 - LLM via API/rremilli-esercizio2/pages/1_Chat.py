import streamlit as st
import requests
import time

# Configurazione pagina
st.set_page_config(page_title="Chat - Chat App", layout="centered")

# Inizializzazione stato sessione
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "endpoint" not in st.session_state:
    st.session_state.endpoint = ""
if "deployment_name" not in st.session_state:
    st.session_state.deployment_name = "gpt-35-turbo"
if "api_version" not in st.session_state:
    st.session_state.api_version = "2023-05-15"

# Verifica autenticazione
if not st.session_state.authenticated:
    st.warning("Devi autenticarti prima di accedere alla chat.")
    st.button("Vai alla pagina di login", on_click=lambda: st.switch_page("streamlit_app.py"))
    st.stop()  # Interrompe l'esecuzione del resto della pagina

# Funzione per chiamare Azure OpenAI
def call_azure_openai_completions(prompt, max_tokens=100):
    completions_endpoint = f"{st.session_state.endpoint}/openai/deployments/{st.session_state.deployment_name}/completions?api-version={st.session_state.api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": st.session_state.api_key
    }
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "top_p": 0.95
    }
    try:
        response = requests.post(
            url=completions_endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Errore: {response.text}")
            return None
    except Exception as e:
        print(f"Errore durante la chiamata API: {e}")
        return None

# Pagina di chat
st.title("Chat")

# Pulsante per disconnettersi
if st.button("Disconnetti", type="secondary"):
    st.session_state.authenticated = False
    st.switch_page("streamlit_app.py")

# Mostra messaggi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utente
if prompt := st.chat_input("Messaggio..."):
    # Aggiungi messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Risposta assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Prompt contestualizzato
            context_prompt = f"Sei un assistente che fornisce risposte brevi e pertinenti. Rispondi a: {prompt}"
            response = call_azure_openai_completions(context_prompt, max_tokens=100)
            
            if response and "choices" in response and len(response["choices"]) > 0:
                assistant_response = response["choices"][0]["text"].strip()
                
                # Effetto digitazione
                text = ""
                for char in assistant_response:
                    text += char
                    message_placeholder.markdown(text + "▌")
                    time.sleep(0.01)
                message_placeholder.markdown(assistant_response)
            else:
                message_placeholder.markdown("Nessuna risposta.")
        except Exception as e:
            message_placeholder.markdown(f"Errore: {str(e)}")
        
    # Aggiorna cronologia
    if 'assistant_response' in locals():
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
