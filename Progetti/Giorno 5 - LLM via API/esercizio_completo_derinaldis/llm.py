import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
 
load_dotenv()

# deploment e version presi dal .env mentre key e endpoint da far inserire all'utente
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_OPENAI_KEY = None
AZURE_OPENAI_ENDPOINT = None
client = None

def get_client(endpoint: str, api_key: str):

    AZURE_OPENAI_KEY = api_key
    AZURE_OPENAI_ENDPOINT = endpoint

    client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    )

    return client

def validate_key_endpoint(key: str, endpoint: str) -> bool:

    try:
        client = get_client(endpoint, key)
        return True
    except Exception as e:
        print(f"Error validating key and endpoint: {e}")
        return False

# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")


# client = AzureOpenAI(
#     api_version=AZURE_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_KEY,
# )

def ask_openai(user_text: str, client) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Ans",
            },
            {
                "role": "user",
                "content": user_text,
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=AZURE_OPENAI_DEPLOYMENT
    )
    return response.choices[0].message.content
 
# response = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant. Ans",
#         },
#         {
#             "role": "user",
#             "content": "I am going to Paris, what should I see?",
#         }
#     ],
#     max_tokens=4096,
#     temperature=1.0,
#     top_p=1.0,
#     model=AZURE_OPENAI_DEPLOYMENT
# )

# @retry(
#     wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s, max 10s
#     stop=stop_after_attempt(5)  # massimo 5 tentativi
# )
# def ask():
#     return client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant. Ans",
#             },
#             {
#                 "role": "user",
#                 "content": "I am going to Paris, what should I see?",
#             }
#         ],
#         max_tokens=4096,
#         temperature=1.0,
#         top_p=1.0,
#         model=AZURE_OPENAI_DEPLOYMENT
# )

# response = ask()
 
#print(response.choices[0].message.content)