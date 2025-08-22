import streamlit as st
from typing import List, Dict
from azure_ai_client import AzureAIClient

def initialize_session_state():
    """Initialize session state variables for chat history"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "azure_client" not in st.session_state:
        st.session_state.azure_client = AzureAIClient()

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def chat_interface():
    initialize_session_state()

    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Settings"])
    
    with tab1:
        st.title("🤖 Azure AI Chat")
        st.caption("A minimal ChatGPT-like interface powered by Azure AI")
        display_chat_history()

        # Check if we're waiting for a response
        if st.session_state.get("waiting_for_response", False):
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Generate assistant response
                    prompt = st.session_state.messages[-1]["content"]
                    try:
                        response = st.session_state.azure_client.chat_completion(prompt, st.session_state.messages[:-1])
                    except Exception as e:
                        response = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.waiting_for_response = False
                    st.rerun()
            return

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.waiting_for_response = True
            st.rerun()
    
    with tab2:
        st.title("⚙️ Settings")
        st.caption("Configure your chat experience")
        
        # Add settings options here
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 1, 4000, 1000)
        
        st.subheader("Display Settings")
        theme = st.selectbox("Theme", ["Default", "Dark", "Light"])
        show_timestamps = st.checkbox("Show message timestamps")
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    chat_interface()