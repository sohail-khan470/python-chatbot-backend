from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import uvicorn
import os
import uuid
import json
from typing import List

from app.models import QueryRequest, QueryResponse, IngestResponse, HealthResponse
from app.rag_pipeline import RAGPipeline
from app.vector_store import VectorStoreManager
from app.document_processor import DocumentProcessor

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_pipeline = RAGPipeline()
vector_store = VectorStoreManager()
document_processor = DocumentProcessor()

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        vector_store_ready=True
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the vector store"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = file.filename.split('.')[-1]
    temp_filename = f"{uuid.uuid4()}.{file_extension}"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        documents = await document_processor.process_uploaded_file(temp_filepath, file.filename)
        chunk_ids = vector_store.add_documents(documents)
        
        os.remove(temp_filepath)
        
        return IngestResponse(
            message=f"Successfully processed {file.filename}",
            document_id=str(chunk_ids[0]) if chunk_ids else "unknown",
            chunks_processed=len(chunk_ids)
        )
        
    except Exception as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system (non-streaming)"""
    try:
        result = await rag_pipeline.generate_response(
            question=request.question,
            conversation_history=request.conversation_history
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/query-stream")
async def query_rag_stream(request: QueryRequest):
    """Streaming query endpoint"""
    try:
        # First, retrieve sources for the response
        relevant_docs = vector_store.similarity_search(request.question)
        sources = [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
        
        async def generate():
            # Send sources first
            yield json.dumps({
                "type": "sources",
                "data": sources
            }) + "\n"
            
            # Ensure sources are sent immediately
            await asyncio.sleep(0.01)
            
            # Stream the response tokens
            async for token in rag_pipeline.generate_streaming_response(
                question=request.question,
                conversation_history=request.conversation_history
            ):
                yield json.dumps({
                    "type": "token",
                    "data": token
                }) + "\n"
                await asyncio.sleep(0.001)  # Small delay for proper streaming
            
            # Send end signal
            yield json.dumps({
                "type": "end",
                "data": "stream_complete"
            }) + "\n"
        
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        async def error_stream():
            yield json.dumps({
                "type": "error",
                "data": f"Error: {str(e)}"
            }) + "\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="application/x-ndjson"
        )

@app.get("/stats")
async def get_stats():
    """Get vector store statistics"""
    return vector_store.get_collection_stats()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )