# store_index.py - One-time script to process PDF, create embeddings, and store in ChromaDB

import os
import sys
from dotenv import load_dotenv
from src.helper import (
    load_pdf, 
    split_documents, 
    create_embeddings, 
    create_vector_store
)

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it in your .env file.")
    sys.exit(1)

def main():
    # Define paths
    pdf_path = os.path.join("data", "medical_book.pdf")
    persist_directory = os.path.join("data", "chroma_db")
    
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    print(f"Processing PDF: {pdf_path}")
    
    # Load PDF
    try:
        documents = load_pdf(pdf_path)
        print(f"Loaded {len(documents)} pages from PDF")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        sys.exit(1)
    
    # Split documents into chunks
    try:
        chunks = split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks")
    except Exception as e:
        print(f"Error splitting documents: {e}")
        sys.exit(1)
    
    # Create embeddings
    try:
        embeddings = create_embeddings()
        print("Created embeddings model")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        sys.exit(1)
    
    # Create and persist vector store
    try:
        vectordb = create_vector_store(chunks, embeddings, persist_directory)
        print(f"Created and persisted vector store at {persist_directory}")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        sys.exit(1)
    
    print("\nIndexing completed successfully!")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Vector store location: {persist_directory}")
    print("\nYou can now run the app.py to start the chatbot.")

if __name__ == "__main__":
    main()