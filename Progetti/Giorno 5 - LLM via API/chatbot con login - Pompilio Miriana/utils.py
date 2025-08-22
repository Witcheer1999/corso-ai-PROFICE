import os
from openai import AzureOpenAI

api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def test_connection(endpoint, api_key):
    """Testa la connessione Azure OpenAI"""
    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        response = client.completions.create(
            model=deployment,
            prompt="Test",
            max_tokens=1
        )
        return True, client
    except Exception as e:
        return False, str(e)
