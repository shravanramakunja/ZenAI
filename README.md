# ZEN AI (RAG-based)

A domain-specific medical chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate, document-grounded answers to medical questions using a custom internal medical reference document.

## Project Overview

This project implements a medical chatbot that:

1. Uses a custom medical reference document (`medical_book.pdf`) as its knowledge base
2. Processes and indexes the document using vector embeddings
3. Retrieves relevant information based on user queries
4. Generates accurate, context-aware responses using Google's Gemini model
5. Presents information through a user-friendly Streamlit interface

## Tech Stack

- **Language Model**: Gemini API (via Google Generative AI)
- **Database**: ChromaDB (local vector store for embeddings)
- **Document**: `medical_book.pdf` (source knowledge base)
- **Embedding Model**: `GoogleGenerativeAIEmbeddings`
- **Framework**: Streamlit (for building interactive chatbot UI)
- **Vector Search**: Cosine similarity-based top-k retrieval
- **RAG Pipeline**: PDF → Text chunks → Embeddings → Chroma → Retrieval → LLM

##  Project Structure

```
health-chatbot/
│
├── data/
│   ├── medical_book.pdf              # Source knowledge base
│   └── chroma_db/                    # Generated vector database (after indexing)
│
├── src/
│   ├── __init__.py
│   ├── helper.py                     # PDF loader, chunk splitter, embeddings, retriever
│   └── prompt.py                     # System prompts for Gemini
│
├── store_index.py                    # One-time script: chunk + embed + store in ChromaDB
├── app.py                            # Streamlit app for user interaction
├── requirements.txt                  # Project dependencies
├── .env                              # API keys and configs
│
├── static/
│   └── style.css                     # Custom styling
│
├── templates/
│   └── chat.html                     # HTML template (optional, as Streamlit handles UI)
│
└── README.md                         # Project guide
```

##  Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Google API key for Gemini

### Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd medical-chatbot
```

2. **Create a virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root with your Google API key:

```
GOOGLE_API_KEY=your_api_key_here
```

5. **Place your medical reference document**

Ensure your `medical_book.pdf` is in the `data/` directory.

6. **Process and index the document**

```bash
python store_index.py
```

This will:
- Load the PDF
- Split it into chunks
- Create embeddings
- Store them in ChromaDB

##  Usage

1. **Start the Streamlit app**

```bash
streamlit run app.py
```

2. **Access the chatbot**

Open your browser and go to `http://localhost:8501`

3. **Ask medical questions**

The chatbot will:
- Retrieve relevant information from the medical reference
- Generate responses based only on that information
- Inform you if the answer isn't in the reference material

##  How It Works

### RAG Pipeline

1. **Document Processing**:
   - The PDF is loaded and split into manageable chunks
   - Each chunk is processed to maintain context and readability

2. **Embedding Generation**:
   - Google's embedding model converts text chunks into vector representations
   - These vectors capture the semantic meaning of the text

3. **Vector Storage**:
   - ChromaDB stores these vectors efficiently for quick retrieval
   - The database maintains the relationship between vectors and original text

4. **Query Processing**:
   - User questions are converted to the same vector space
   - Similarity search finds the most relevant document chunks

5. **Response Generation**:
   - Retrieved chunks are sent to Gemini along with the user query
   - A carefully crafted system prompt ensures responses use only the provided context
   - The model generates a natural language response grounded in the reference material

##  Notes

- This chatbot only provides information from the reference document and does not offer medical advice
- The quality of responses depends on the content and coverage of the medical reference document
- For production use, consider implementing additional security measures and user authentication

##  Privacy and Security

- All processing happens locally except for API calls to Google's Gemini
- No user data or queries are stored permanently
- The medical reference document stays within your system

