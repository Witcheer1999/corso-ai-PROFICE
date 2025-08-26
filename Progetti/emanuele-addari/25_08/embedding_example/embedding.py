from urllib import response
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class Ciatgpt:
    def __init__(self, api_key, conn_str):
        self.api_key = api_key
        self.connection_string = conn_str
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            api_key=self.api_key,
            azure_endpoint=self.connection_string
        )

    def embedding_client(self):
        return self.client.embeddings.create(
            model=self.deployment,
            input="hello world"
        )
    def answer(self):
        response = self.embedding_client()
        return response.data[0].embedding

if __name__ == "__main__":

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    conn_str = os.getenv("AZURE_OPENAI_ENDPOINT")

    chat_gpt = Ciatgpt(api_key, conn_str)
    response = chat_gpt.answer()
    print(len(response))