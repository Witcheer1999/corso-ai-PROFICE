import streamlit as st
from auth_utils import init_session_state, build_client, mask_key

st.set_page_config(page_title="Login Azure OpenAI", page_icon="🔐")

init_session_state()

st.title("🔐 Login Azure OpenAI")
st.write("Inserisci le credenziali. Non vengono salvate su disco. Dopo l'accesso vai alla pagina 'Chat'.")

with st.form("login_form"):
    endpoint_in = st.text_input("Endpoint Azure", value=st.session_state.endpoint, placeholder="https://<nome-risorsa>.openai.azure.com")
    deployment_in = st.text_input("Deployment / Model name", value=st.session_state.deployment, placeholder="es. gpt-4o-mini")
    api_key_in = st.text_input("API Key", value=st.session_state.api_key, type="password")
    submitted = st.form_submit_button("Connetti")

if submitted:
    if not (endpoint_in and api_key_in and deployment_in):
        st.error("Compila tutti i campi.")
    else:
        st.session_state.endpoint = endpoint_in.strip()
        st.session_state.api_key = api_key_in.strip()
        st.session_state.deployment = deployment_in.strip()
        try:
            tmp_client = build_client(st.session_state.endpoint, st.session_state.api_key)
            if tmp_client is None:
                st.error("Impossibile creare il client. Verifica i dati.")
            else:
                st.session_state.authenticated = True
                st.success("Connessione riuscita! Vai alla pagina Chat.")
        except Exception as e:
            st.error(f"Errore connessione: {e}")

if st.session_state.get("authenticated"):
    st.info(f"Autenticato. Endpoint: {st.session_state.endpoint}\nAPI Key: {mask_key(st.session_state.api_key)}")
    st.page_link("pages/1_💬_Chat.py", label="➡️ Vai alla Chat", icon="💬")
else:
    st.caption("Non sei autenticato.")
