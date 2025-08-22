import streamlit as st
from typing import List, Dict
from azure_ai_client import AzureAIClient

def initialize_session_state():
    """Initialize session state variables for chat history"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "azure_client" not in st.session_state:
        st.session_state.azure_client = None

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def chat_interface(azure_client: AzureAIClient):
    """Main chat interface component"""
    
    # Initialize session state
    initialize_session_state()
    
    # Store the client in session state
    if st.session_state.azure_client is None:
        st.session_state.azure_client = azure_client
    
    st.title("🤖 Azure AI Chat")
    st.caption("A minimal ChatGPT-like interface powered by Azure AI")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = azure_client.chat_completion(prompt, st.session_state.messages[:-1])
                    st.markdown(response)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

def chat_sidebar():
    """Sidebar with chat controls"""
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Display conversation stats
        st.subheader("📊 Stats")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        st.metric("Total Messages", total_messages)
        st.metric("Your Messages", user_messages)
        st.metric("AI Responses", assistant_messages)