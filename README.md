# RAG Chatbot Backend

A FastAPI-based backend service for a Retrieval-Augmented Generation (RAG) chatbot that processes documents and provides conversational AI responses.

## Overview

This backend implements a complete RAG pipeline that:

- Processes and ingests documents (PDF, TXT, CSV)
- Stores document embeddings in a vector database (ChromaDB)
- Provides streaming chat responses using Ollama models
- Maintains conversation history for context-aware responses

## Architecture

### Core Components

- **FastAPI Application** (`main.py`): Main API server with endpoints for document ingestion, querying, and health checks
- **Document Processor** (`document_processor.py`): Handles file parsing and preprocessing for different document types
- **Vector Store Manager** (`vector_store.py`): Manages document embeddings and similarity search using ChromaDB
- **RAG Pipeline** (`rag_pipeline.py`): Orchestrates the retrieval and generation process
- **Configuration** (`config.py`): Centralized settings for models, chunking, and API parameters

### Key Features

- **Multi-format Document Support**: PDF, TXT, and CSV files
- **Streaming Responses**: Real-time token-by-token response generation
- **Conversation History**: Context-aware responses using chat history
- **Vector Search**: Semantic similarity search for relevant document retrieval
- **Health Monitoring**: API health checks and vector store statistics

## API Endpoints

### Document Management

- `POST /ingest` - Upload and process documents
- `GET /stats` - Get vector store statistics

### Chat Interface

- `POST /query` - Non-streaming query (synchronous)
- `POST /query-stream` - Streaming query with real-time responses

### System

- `GET /` - Health check endpoint

## Data Flow

1. **Document Ingestion**:

   - File uploaded via `/ingest` endpoint
   - Document processed based on file type (PDF/TXT/CSV)
   - Text split into chunks with overlap
   - Embeddings generated and stored in ChromaDB

2. **Query Processing**:
   - User question received via `/query-stream`
   - Relevant documents retrieved via similarity search
   - Context built from retrieved documents
   - Prompt constructed with context and conversation history
   - Response streamed token-by-token from Ollama model

## Configuration

Key settings in `config.py`:

- **LLM Model**: Currently set to "phi3:latest" (change to other Ollama models as needed)
- **Embedding Model**: "sentence-transformers/all-MiniLM-L6-v2"
- **Chunk Settings**: 1000 characters with 200 character overlap
- **Vector Search**: Top 3 similar documents retrieved

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `langchain`: LLM framework and document processing
- `chromadb`: Vector database
- `sentence-transformers`: Embedding generation
- `ollama`: Local LLM inference
- `pypdf`: PDF processing

## Installation & Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama**:

   - Download from https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull phi3:latest
     # or other models: llama3.2, mistral, codellama, etc.
     ```

3. **Run the Server**:
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Usage

### Document Upload

```python
import requests

files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/ingest', files=files)
print(response.json())
```

### Chat Query

```python
import requests

data = {
    "question": "What is the main topic?",
    "conversation_history": []
}

response = requests.post('http://localhost:8000/query-stream', json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── models.py         # Pydantic models
│   ├── document_processor.py  # File processing logic
│   ├── vector_store.py   # ChromaDB management
│   └── rag_pipeline.py   # RAG orchestration
├── data/
│   ├── uploads/          # Temporary uploaded files
│   └── vector_store/     # ChromaDB persistence
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Development

- **Hot Reload**: Server restarts automatically on code changes
- **API Documentation**: Available at `http://localhost:8000/docs` (Swagger UI)
- **Health Check**: `GET /` returns system status
- **Vector Store Stats**: `GET /stats` shows document count and collection info

## Troubleshooting

- **Model Issues**: Ensure Ollama is running and models are pulled
- **Port Conflicts**: Change port in config or kill conflicting processes
- **Memory Issues**: Reduce chunk size or similarity search results
- **File Upload Errors**: Check file size limits (10MB default) and supported formats

## Performance Considerations

- **Chunk Size**: Balance between context richness and search precision
- **Embedding Model**: CPU-based model; consider GPU acceleration for larger deployments
- **Vector Search**: Top-K results affect response quality and speed
- **Streaming**: Reduces perceived latency for long responses
