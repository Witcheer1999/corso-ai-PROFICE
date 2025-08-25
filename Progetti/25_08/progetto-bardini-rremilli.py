import streamlit as st
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time
from dotenv import load_dotenv

# Load environment variables if present
load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
# Shared endpoint
if "azure_endpoint" not in st.session_state:
    st.session_state.azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
# Chat model configuration
if "azure_chat_api_key" not in st.session_state:
    st.session_state.azure_chat_api_key = os.getenv("AZURE_CHAT_API_KEY", "")
if "azure_chat_api_version" not in st.session_state:
    st.session_state.azure_chat_api_version = os.getenv("AZURE_CHAT_API_VERSION", "2025-01-01-preview")
if "azure_chat_deployment" not in st.session_state:
    st.session_state.azure_chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "")
# Embedding model configuration
if "azure_embedding_api_key" not in st.session_state:
    st.session_state.azure_embedding_api_key = os.getenv("AZURE_EMBEDDING_API_KEY", "")
if "azure_embedding_api_version" not in st.session_state:
    st.session_state.azure_embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
if "azure_embedding_deployment" not in st.session_state:
    st.session_state.azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")

# Define the pages
def login_page():
    st.title("Azure OpenAI Configuration")
    
    st.markdown("""
    ## Configure Azure OpenAI API
    
    Please enter your Azure OpenAI API credentials to use the RAG Chat Assistant.
    You can configure different API keys and versions for chat and embedding models.
    """)
    
    with st.form("azure_config_form"):
        # Shared endpoint
        st.subheader("Shared Configuration")
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint (Shared)",
            placeholder="https://your-resource-name.openai.azure.com",
            value=st.session_state.azure_endpoint,
            help="This endpoint will be used for both chat and embedding models"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Chat Model Configuration")
            azure_chat_api_key = st.text_input(
                "Chat Model API Key", 
                type="password",
                value=st.session_state.azure_chat_api_key
            )
            
            azure_chat_api_version = st.text_input(
                "Chat Model API Version",
                value=st.session_state.azure_chat_api_version
            )
            
            azure_chat_deployment = st.text_input(
                "Chat Model Deployment Name",
                placeholder="e.g., gpt-4-deployment",
                value=st.session_state.azure_chat_deployment
            )
        
        with col2:
            st.subheader("Embedding Model Configuration")
            azure_embedding_api_key = st.text_input(
                "Embedding Model API Key", 
                type="password",
                value=st.session_state.azure_embedding_api_key
            )
            
            azure_embedding_api_version = st.text_input(
                "Embedding Model API Version",
                value=st.session_state.azure_embedding_api_version
            )
            
            azure_embedding_deployment = st.text_input(
                "Embedding Model Deployment Name",
                placeholder="e.g., embedding-ada-002-deployment",
                value=st.session_state.azure_embedding_deployment
            )
        
        submit_button = st.form_submit_button("Save Configuration")
        
        if submit_button:
            if not (azure_endpoint and azure_chat_api_key and azure_chat_deployment and 
                    azure_embedding_api_key and azure_embedding_deployment):
                st.error("Please fill in all the required fields.")
            else:
                # Save the configuration to session state
                st.session_state.azure_endpoint = azure_endpoint
                st.session_state.azure_chat_api_key = azure_chat_api_key
                st.session_state.azure_chat_api_version = azure_chat_api_version
                st.session_state.azure_chat_deployment = azure_chat_deployment
                st.session_state.azure_embedding_api_key = azure_embedding_api_key
                st.session_state.azure_embedding_api_version = azure_embedding_api_version
                st.session_state.azure_embedding_deployment = azure_embedding_deployment
                
                st.session_state.logged_in = True
                st.success("Configuration saved successfully!")
                st.rerun()

def main_page():
    st.title("RAG Chat Assistant")
    
    # Sidebar for settings
    with st.sidebar:
        st.title("Settings")
        
        st.write(f"**Azure Endpoint:** {st.session_state.azure_endpoint.split('.')[-2].split('/')[-1]}")
        st.write(f"**Chat Model:** {st.session_state.azure_chat_deployment}")
        st.write(f"**Embedding Model:** {st.session_state.azure_embedding_deployment}")
        
        st.info("Note: The o4-mini model uses default temperature (1.0) and doesn't support custom temperature values.")
        
        if st.session_state.documents_processed:
            st.success("Documents processed! You can now chat with your documents.")
        
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.conversation = None
            st.rerun()
        
        if st.button("Change Azure Configuration"):
            st.session_state.logged_in = False
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for user queries
    if prompt := st.chat_input("Ask me something about your documents..."):
        if not st.session_state.documents_processed:
            st.error("Please upload and process documents first in the 'Upload Documents' page.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get response from conversational chain
            result = st.session_state.conversation({"question": prompt})
            response = result['answer']
            
            # Simulate streaming effect
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.write(full_response + "▌")
            
            message_placeholder.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def upload_page():
    st.title("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Create a temporary directory to save the uploaded files
                temp_dir = tempfile.mkdtemp()
                st.session_state.temp_dir = temp_dir
                
                # Save and load the documents
                docs = []
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load documents based on file type
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(temp_file_path)
                    elif file.name.endswith(".docx"):
                        loader = Docx2txtLoader(temp_file_path)
                    elif file.name.endswith(".txt"):
                        loader = TextLoader(temp_file_path)
                    elif file.name.endswith(".csv"):
                        loader = CSVLoader(temp_file_path)
                    elif file.name.endswith((".xlsx", ".xls")):
                        loader = UnstructuredExcelLoader(temp_file_path)
                    else:
                        st.error(f"Unsupported file format: {file.name}")
                        continue
                    
                    docs.extend(loader.load())
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(docs)
                
                # Create embeddings using Azure OpenAI
                embeddings = AzureOpenAIEmbeddings(
                    deployment=st.session_state.azure_embedding_deployment,
                    model="text-embedding-3-small",
                    api_key=st.session_state.azure_embedding_api_key,
                    azure_endpoint=st.session_state.azure_endpoint,
                    api_version=st.session_state.azure_embedding_api_version
                )
                
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Create conversation chain with Azure OpenAI
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                
                # Initialize the conversation chain with Azure OpenAI
                llm = AzureChatOpenAI(
                    deployment_name=st.session_state.azure_chat_deployment,
                    api_key=st.session_state.azure_chat_api_key,
                    azure_endpoint=st.session_state.azure_endpoint,
                    api_version=st.session_state.azure_chat_api_version,
                    streaming=True
                )
                
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    memory=memory,
                    return_source_documents=True
                )
                
                st.session_state.documents_processed = True
                st.success(f"Successfully processed {len(chunks)} document chunks!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your Azure OpenAI configuration and try again.")

# Determine which page to show
def show_page():
    if not st.session_state.logged_in:
        login_page()
    else:
        # Navigation after login
        page_names_to_funcs = {
            "Chat": main_page,
            "Upload Documents": upload_page,
        }
        
        selected_page = st.sidebar.radio("Navigation", page_names_to_funcs.keys())
        page_names_to_funcs[selected_page]()

# Run the app
show_page()
