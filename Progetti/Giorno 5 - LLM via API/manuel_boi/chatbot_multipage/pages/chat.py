import streamlit as st

st.set_page_config(page_title="Chat Azure", layout="centered")

if "client" not in st.session_state:
    st.warning("⚠️ Devi prima connetterti. Torna alla pagina principale.")
    st.stop()

st.title("Chatbot GPT-4o-mini")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Scrivi qui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        stream = st.session_state.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=st.session_state.deployment,
            temperature=1.0,
            max_tokens=4096,
            stream=True
        )

        for chunk in stream:
            if (
                chunk.choices and
                chunk.choices[0].delta and
                hasattr(chunk.choices[0].delta, "content") and
                chunk.choices[0].delta.content
            ):
                streamed_text += chunk.choices[0].delta.content
                placeholder.markdown(streamed_text)

    st.session_state.messages.append({"role": "assistant", "content": streamed_text})
