import streamlit as st
from openai import AzureOpenAI
from utils.utils import validate_endpoint_and_key, validate_deployment


st.set_page_config(page_title="Setup", page_icon="🔐")
st.title("🔐 Azure OpenAI Setup")

api_key = st.text_input("API key", type="password")
endpoint = st.text_input("Azure OpenAI endpoint")
api_version = st.text_input("API version")
deployment_name = st.text_input("Deployment name")


if st.button("Test & continue"):
  # Build client from user inputs
  try:
    client = AzureOpenAI(
      api_version=api_version,
      azure_endpoint=endpoint,
      api_key=api_key,
      )
  except Exception as e:
    st.error(f"❌ Impossibile creare il client: {e}")
    st.stop()

  # 1) Check endpoint/key/version
  ok, reason = validate_endpoint_and_key(client)
  if not ok:
    st.error(f"❌ Connessione fallita. Controlla key / endpoint / API version.\n\n{reason}")
    st.stop()

  # 2) Check deployment existence/access
  ok_dep, reason_dep = validate_deployment(client, deployment_name)
  if not ok_dep:
      st.error(f"❌ Deployment non valido.\n\n{reason_dep}")
      st.stop()

  st.session_state.update({
    "subscription_key": api_key,
    "azure_endpoint": endpoint,
    "api_version": api_version,
    "deployment_name": deployment_name,
    "auth_ok": True,
  })
  st.success("✅ Connessione OK!")
  try:
    st.switch_page("pages/chat_page.py")
  except Exception:
    st.page_link("pages/chat_page.py", label="Vai alla chat →")