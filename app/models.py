from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = []

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class IngestResponse(BaseModel):
    message: str
    document_id: str
    chunks_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_store_ready: bool