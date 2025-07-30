# helper.py - Functions for PDF processing, chunking, embedding, and retrieval

import os
import asyncio
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_pdf(pdf_path: str) -> List[Any]:
    """
    Load a PDF file and extract its content as documents.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of document pages
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """
    Split documents into smaller chunks for better processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create Google Generative AI embeddings.
    
    Returns:
        GoogleGenerativeAIEmbeddings object
    """
    # Ensure event loop is available
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return embeddings

def create_vector_store(chunks: List[Any], embeddings: Any, persist_directory: str) -> Chroma:
    """
    Create a vector store from document chunks using the provided embeddings.
    
    Args:
        chunks: List of document chunks
        embeddings: Embeddings object
        persist_directory: Directory to persist the vector store
        
    Returns:
        Chroma vector store
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # Note: Since Chroma 0.4.x, docs are automatically persisted
    return vectordb

def load_vector_store(embeddings: Any, persist_directory: str) -> Chroma:
    """
    Load an existing vector store from the persist directory.
    
    Args:
        embeddings: Embeddings object
        persist_directory: Directory where the vector store is persisted
        
    Returns:
        Chroma vector store
    """
    # Ensure event loop is available
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Check if the persist directory exists and has content
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector store directory not found: {persist_directory}")
    
    # Check for essential ChromaDB files
    chroma_db_file = os.path.join(persist_directory, "chroma.sqlite3")
    if not os.path.exists(chroma_db_file):
        raise FileNotFoundError(f"ChromaDB file not found: {chroma_db_file}")
    
    try:
        # Try to load with the new approach first
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="medical_documents"  # Try with explicit collection name
        )
        
        # Test if the collection has documents
        collection_count = vectordb._collection.count()
        if collection_count == 0:
            # Try without collection name
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            collection_count = vectordb._collection.count()
            
        print(f"Vector store loaded successfully with {collection_count} documents")
        return vectordb
        
    except Exception as e:
        print(f"Error loading vector store: {e}")
        # Fallback: try to recreate from the PDF if it exists
        pdf_path = os.path.join("data", "medical_book.pdf")
        if os.path.exists(pdf_path):
            print("Attempting to recreate vector store from PDF...")
            return recreate_vector_store_from_pdf(pdf_path, embeddings, persist_directory)
        else:
            raise Exception(f"Failed to load vector store and PDF not found: {e}")

def recreate_vector_store_from_pdf(pdf_path: str, embeddings: Any, persist_directory: str) -> Chroma:
    """
    Recreate vector store from PDF file as fallback.
    """
    print("Recreating vector store from PDF...")
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    print(f"Creating vector store with {len(chunks)} chunks...")
    vectordb = create_vector_store(chunks, embeddings, persist_directory)
    return vectordb

def retrieve_relevant_chunks(query: str, vectordb: Chroma, k: int = 4) -> List[str]:
    """
    Retrieve the most relevant document chunks for a given query.
    
    Args:
        query: User query
        vectordb: Vector store to search in
        k: Number of chunks to retrieve
        
    Returns:
        List of relevant document chunks as strings
    """
    docs = vectordb.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]