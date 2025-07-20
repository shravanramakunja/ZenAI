# app.py - Streamlit app for medical chatbot with RAG implementation

import os
import sys
import streamlit as st
import google.genai as genai
from dotenv import load_dotenv
from src.helper import create_embeddings, load_vector_store, retrieve_relevant_chunks
from src.prompt import get_system_prompt, get_query_prompt

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Error: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
    st.stop()

# Initialize the client
genai_client = genai.Client(api_key=api_key)

# Set page configuration
st.set_page_config(
    page_title="Zen AI",
    layout="centered"
)

# Define paths
persist_directory = os.path.join("data", "chroma_db")

# Check if vector store exists
if not os.path.exists(persist_directory):
    st.error(
        "Error: Vector store not found. Please run store_index.py first to create the vector store."
        "\n\nRun: python store_index.py"
    )
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load embeddings and vector store
@st.cache_resource
def load_resources():
    try:
        embeddings = create_embeddings()
        vectordb = load_vector_store(embeddings, persist_directory)
        return embeddings, vectordb
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

embeddings, vectordb = load_resources()

# Initialize Gemini model
@st.cache_resource
def load_model():
    try:
        # Return the client itself, as we'll use client.models.generate_content directly
        return genai_client
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Function to generate response
def generate_response(query):
    # Retrieve relevant chunks from the vector store
    relevant_chunks = retrieve_relevant_chunks(query, vectordb, k=4)
    
    # Join the chunks into a single context string
    context = "\n\n".join(relevant_chunks)
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
    # Format query with context
    formatted_prompt = get_query_prompt(query, context)
    
    # Generate response using Gemini
    config = genai.types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        system_instruction=system_prompt,
        safety_settings=[
            genai.types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            genai.types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            genai.types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            genai.types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
        ]
    )
    
    response = model.models.generate_content(
        model="gemini-2.0-flash",
        contents=[genai.types.Content(parts=[genai.types.Part(text=formatted_prompt)])],
        config=config
    )
    
    return response.text

# App title and description
st.title("Zen AI ")
st.markdown(
    """
    Ask medical questions and get answers based on the medical reference material.
    This assistant only provides information from the reference document and does not offer medical advice.
    """
)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response with a spinner while generating
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(query)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown(
        """
        This Medical AI Assistant uses RAG (Retrieval-Augmented Generation) to provide accurate 
        information from a medical reference document.
        
        **How it works:**
        1. Your question is processed to find relevant information in the medical reference
        2. The AI generates an answer based ONLY on that information
        3. If the answer isn't in the reference, the AI will let you know
        
        **Note:** This is for educational purposes only and not a substitute for professional medical advice.
        """
    )
    
    st.divider()
    
    st.markdown(
        """
        **Sample Questions:**
        - What are the symptoms of diabetes?
        - How is hypertension diagnosed?
        - What treatments are available for asthma?
        - What are the risk factors for heart disease?
        """
    )