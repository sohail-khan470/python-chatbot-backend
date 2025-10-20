import os
from typing import Optional

class Settings:
    # Vector Store
    VECTOR_STORE_PATH = "./data/vector_store"
    CHROMA_COLLECTION_NAME = "rag_documents"
    
    # Model Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3.1:latest"  # or "llama2", "codellama" - make sure you have these in Ollama
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_TOP_K = 3
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
settings = Settings()