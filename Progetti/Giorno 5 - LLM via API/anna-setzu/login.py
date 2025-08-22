from openai import AzureOpenAI, APIConnectionError, RateLimitError, APIStatusError
import streamlit as st

st.title('Connessione a LLM Azure')
st.caption('Inserisci Endpoint e API key')

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "client" not in st.session_state:
    st.session_state.client = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_model" not in st.session_state:
    st.session_state.openai_model = st.secrets.get("AZURE_OPENAI_DEPLOYMENT", "")

def validate_credentials(endpoint: str, api_key: str, deployment: str):
    try:
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-05-01-preview",
        )
        _ = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a health check."},
                {"role": "user", "content": "ping"},
            ],
            max_tokens=1,
            temperature=0,
        )
        return client, None
    except APIConnectionError:
        return None, "Impossibile connettersi all'endpoint. Controlla l'URL o la rete."
    except RateLimitError:
        return None, "Rate limit raggiunto per questa chiave/risorsa."
    except APIStatusError as e:
        code = getattr(e, "status_code", "N/A")
        return None, f"Errore API ({code}). Verifica chiave, endpoint e permessi del deployment."
    except Exception as e:
        return None, f"Errore inatteso: {e}"

with st.form("credentials_form"):
    endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        value=st.session_state.get("entered_endpoint", ""),
        placeholder="https://<resource-name>.openai.azure.com",
    )
    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("entered_key", ""),
        type="password",
        placeholder="az-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    )
    deployment = st.text_input(
        "Deployment name",
        value=st.session_state.openai_model,
        placeholder="es. gpt-4o-mini",
        help="Nome del deployment configurato su Azure OpenAI",
    )

    submitted = st.form_submit_button("Connetti")

if submitted:
    if not endpoint or not api_key or not deployment:
        st.error("Compila Endpoint, API Key e Deployment.")
    else:
        st.session_state.entered_endpoint = endpoint
        st.session_state.entered_key = api_key
        st.session_state.openai_model = deployment

        client, err = validate_credentials(endpoint, api_key, deployment)
        if err:
            st.error(err)
            st.session_state.auth_ok = False
            st.session_state.client = None
        else:
            st.success("Connessione riuscita! Reindirizzo alla chat…")
            st.session_state.auth_ok = True
            st.session_state.client = client
            st.switch_page("chat.py")
