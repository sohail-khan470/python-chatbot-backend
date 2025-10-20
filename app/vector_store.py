import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Optional,Dict
from app.config import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load existing vector store"""
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        
        return Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_STORE_PATH
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store after splitting"""
        chunks = self.text_splitter.split_documents(documents)
        return self.vector_store.add_documents(chunks)
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents"""
        if k is None:
            k = settings.SIMILARITY_TOP_K
        return self.vector_store.similarity_search(query, k=k)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        collection = self.vector_store._collection
        if collection:
            return {
                "document_count": collection.count(),
                "collection_name": settings.CHROMA_COLLECTION_NAME
            }
        return {"document_count": 0}