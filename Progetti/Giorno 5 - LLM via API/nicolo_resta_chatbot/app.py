import streamlit as st
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from azure_ai_client import AzureAIClient
from chat_interface import chat_interface, chat_sidebar

def main():
    """Main application entry point"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Azure AI Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        
        # Initialize Azure AI client
        azure_client = AzureAIClient()
        
        # Create layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Main chat interface
            chat_interface(azure_client)
        
        with col2:
            # Sidebar controls
            chat_sidebar()
            
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please check your .env file and ensure all required environment variables are set.")
        
        with st.expander("Required Environment Variables"):
            st.code("""
API_KEY=your_azure_api_key_here
ENDPOINT_URL=https://your_azure_endpoint_url_here
MODEL_NAME=your_model_name_here
            """)
            
    except Exception as e:
        st.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main()