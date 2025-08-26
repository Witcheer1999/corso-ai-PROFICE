from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential

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
        self.conversation_history = []

    def check_keys(self):
        missing_keys = [key for key in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME"] if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")
        if not self.client.models.list():
            raise ValueError("Wrong keys")


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def make_api_call(self):
        return self.client.chat.completions.create(
            messages=self.conversation_history,
            max_completion_tokens=200,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=self.deployment,
            stream=True
        )
    
    def answer(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.make_api_call()
        answer = ""
        for chunk in response:
            if chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        if answer:
            self.conversation_history.append({"role": "assistant", "content": answer})
