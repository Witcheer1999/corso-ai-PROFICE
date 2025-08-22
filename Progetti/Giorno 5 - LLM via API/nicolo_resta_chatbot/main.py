import streamlit as st
from azure_ai_client import AzureAIClient

def test_conn(ep, key, dep) -> str:
    """Ritorna una stringa se tutto ok, alza eccezione se fallisce."""
    client = AzureAIClient(key=key, endpoint=ep, model_name=dep)
    _ = client.generate_response(
        messages=[{"role":"system","content":"pong?"},{"role":"user","content":"ping"}]
    )
    return client

def main():
    st.set_page_config(page_title="Setup Azure OpenAI", page_icon="🔑")
    st.title("🔑 Configura Azure OpenAI")

    # Campi di input
    endpoint = st.text_input("Endpoint Azure OpenAI", value="https://<resource>.openai.azure.com")
    api_key = st.text_input("API Key", type="password")
    # api_version = st.text_input("API Version", value="2024-12-01-preview")
    deployment = st.text_input("Deployment name (model)", value="gpt-4o-mini")

    disabled = not (endpoint and api_key and deployment)
    if st.button("Test & vai alla chat", use_container_width=True, disabled=disabled):
        try:
            st.session_state.azure_client = test_conn(endpoint, api_key, deployment)
            st.success("Connessione OK!")
            st.switch_page("pages/chat_interface.py")
        except Exception as e:
            st.error(f"Connessione fallita: {e}")

if __name__ == "__main__":
    main()