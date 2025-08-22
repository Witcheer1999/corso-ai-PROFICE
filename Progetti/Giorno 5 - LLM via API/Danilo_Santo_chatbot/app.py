import streamlit as st
from openai import AzureOpenAI

st.set_page_config(page_title="Setup Azure OpenAI", page_icon="🔑")
st.title("🔑 Configura Azure OpenAI")

# Campi di input
endpoint = st.text_input("Endpoint Azure OpenAI", value="https://<resource>.openai.azure.com")
api_key = st.text_input("API Key", type="password")
api_version = st.text_input("API Version", value="2024-12-01-preview")
deployment = st.text_input("Deployment name (model)", value="gpt-4o-mini")


def test_conn(ep, key, ver, dep) -> str:
    """Ritorna una stringa se tutto ok, alza eccezione se fallisce."""
    client = AzureOpenAI(api_key=key, api_version=ver, azure_endpoint=ep)
    resp = client.chat.completions.create(
        model=dep,
        messages=[{"role":"system","content":"pong?"},{"role":"user","content":"ping"}],
        max_tokens=5,
        temperature=0
    )
    return resp.choices[0].message.content


disabled = not (endpoint and api_key and api_version and deployment)
if st.button("Test & vai alla chat", use_container_width=True, disabled=disabled):
    try:
        _ = test_conn(endpoint, api_key, api_version, deployment) # Mettere il '_' è una convenzione Python che si usa quando "non interessa il valore di ritorno, serve solo capire se funziona
        # Salvo tutto in sessione per la chat
        st.session_state.azure = {
            "endpoint": endpoint,
            "api_key": api_key,
            "api_version": api_version,
            "deployment": deployment,
        }
        st.success("Connessione OK!")
        # Vai alla pagina chat
        try:
            st.switch_page("pages/02_page_chat.py")
        except Exception:
            st.info("Errore durante lo switch alla page successiva.")
    except Exception as e:
        st.error(f"Connessione fallita: {e}")