# helper.py - Functions for PDF processing, chunking, embedding, and retrieval

import os
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
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
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