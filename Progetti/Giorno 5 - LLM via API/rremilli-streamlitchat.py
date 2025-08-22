import streamlit as st
import time
from azure_openai_simplified import call_azure_openai_completions

# Titolo
st.title("Chat")

# Inizializza la cronologia dei messaggi
if "messages" not in st.session_state:
    st.session_state.messages = []

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
