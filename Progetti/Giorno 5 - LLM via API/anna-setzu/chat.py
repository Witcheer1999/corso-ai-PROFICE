from openai import APIConnectionError, RateLimitError, APIStatusError
import streamlit as st

if not st.session_state.get("auth_ok") or st.session_state.get("client") is None:
    st.switch_page("login.py")

st.title('LLM Azure')

#Storia chat
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Input utente
if prompt := st.chat_input("Scrivi un messaggio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = st.session_state.client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        except APIConnectionError:
            st.error("Connessione persa durante la richiesta. Riprova.")
        except RateLimitError:
            st.error("Rate limit raggiunto. Attendi un momento e riprova.")
        except APIStatusError as e:
            code = getattr(e, "status_code", "N/A")
            st.error(f"Errore API durante la chat ({code}).")
        except Exception as e:
            st.error(f"Errore inatteso: {e}")
