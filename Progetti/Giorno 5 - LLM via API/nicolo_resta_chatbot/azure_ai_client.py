from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict
import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

class AzureAIClient:
    def __init__(self):

        endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
        key = os.getenv("AZURE_INFERENCE_SDK_KEY")

        self.model_name = "gpt-4.1-mini"

        self.client = ChatCompletionsClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using Azure AI Inference"""
        try:
            response = self.client.complete(
                messages=messages,
                model=self.model_name,
                temperature=0.7,
                max_tokens=1000,
                top_p=0.95
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def chat_completion(self, user_message: str, conversation_history: List[Dict[str, str]]) -> str:
        """Complete a chat conversation with context"""
        # Add the new user message to the conversation
        messages = conversation_history + [{"role": "user", "content": user_message}]
        
        return self.generate_response(messages)