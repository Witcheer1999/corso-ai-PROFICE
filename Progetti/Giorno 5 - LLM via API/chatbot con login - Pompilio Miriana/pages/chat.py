import streamlit as st

st.set_page_config(page_title="Chat", page_icon="💬")
st.title("💬 Chatbot")

if "client" not in st.session_state:
    st.warning("⚠️ Devi prima connetterti nella pagina Login.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Scrivi qui..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = st.session_state.client.completions.create(
                    model=st.session_state.deployment,
                    prompt=prompt,
                    max_tokens=500,
                    temperature=1.0
                )
                answer = response.choices[0].text.strip()
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Errore: {e}")
