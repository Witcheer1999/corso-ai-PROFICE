from dotenv import load_dotenv
from openai import AzureOpenAI
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
import os

load_dotenv()

# Load environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = "gpt-4o"

import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://miche-memmadcn-eastus2.cognitiveservices.azure.com/",
    api_key=api_key,
)


def ask(prompt:str) :
    response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful friend.",
        },
        {
            "role": "user",
            "content": f"{prompt}",
        }
    ],
    max_completion_tokens=200,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment
)
    return response
# response = ask()
# print(response.choices[0].message.content)
