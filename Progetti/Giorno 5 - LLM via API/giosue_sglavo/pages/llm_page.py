import streamlit as st

st.title("Chat Bot")

client = st.session_state.get("client", None)

if client is None:
    st.error("Please connect to Azure OpenAI in the Connection Page.")
    st.stop()

deployment_name = "gpt-35-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi della chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Inserisco la barra del prompt
prompt = st.chat_input("Ask anything")

# Gestisco l'input dell'utente
if prompt:

    # Aggiungi il messaggio dell'utente allo stato della sessione
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostra il messaggio dell'utente
    with st.chat_message("user"):
        st.markdown(prompt)

    # Risposta del modello
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                max_tokens=200,
                temperature=1.0,
                top_p=1.0,
                stream=True
            )
            response = st.write_stream(stream)

        except Exception as e:
            response = f"Si è verificato un errore durante la generazione della risposta: {str(e)}"
            st.error(response)

    # Aggiungi la risposta del modello allo stato della sessione
    st.session_state.messages.append({"role": "assistant", "content": response})