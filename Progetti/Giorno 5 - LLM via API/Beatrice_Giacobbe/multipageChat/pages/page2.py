import streamlit as st

def chat():
    st.title("chat")

    #if st.session_state.client is None:
    #    st.error("Devi prima connetterti.")
    #else:
        # Mostra cronologia chat
        # Inizializza lista dei messaggi

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
            stream = st.session_state.client.chat.completions.create(
                model=st.session_state.deployment,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})




chat()
 