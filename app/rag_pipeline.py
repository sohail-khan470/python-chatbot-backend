import ollama
from langchain.schema import Document
from typing import List, Dict, Any, AsyncGenerator
import asyncio
from app.config import settings
from app.vector_store import VectorStoreManager

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm_model = settings.LLM_MODEL
    
    async def generate_streaming_response(self, question: str, conversation_history: List[Dict] = None) -> AsyncGenerator[str, None]:
        """Generate streaming response token by token"""

        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question)

        # Build context from documents
        context = self._build_context(relevant_docs)

        # Build prompt with context and conversation history
        prompt = self._build_prompt(question, context, conversation_history)

        try:
            # Use a queue to handle streaming from sync Ollama in async context
            queue = asyncio.Queue()

            def producer():
                try:
                    stream = ollama.generate(
                        model=self.llm_model,
                        prompt=prompt,
                        stream=True
                    )
                    for chunk in stream:
                        if 'response' in chunk:
                            asyncio.run_coroutine_threadsafe(queue.put(chunk['response']), loop)
                    # Signal end
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(f"I apologize, but I encountered an error: {str(e)}"), loop)

            loop = asyncio.get_event_loop()
            # Run the producer in a thread
            await asyncio.to_thread(producer)

            # Yield chunks as they become available
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

        except Exception as e:
            yield f"I apologize, but I encountered an error: {str(e)}"
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from relevant documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', f'Document {i}')
            context_parts.append(f"Source: {source}\nContent: {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Build the prompt for the LLM"""
        
        # Build conversation history if provided
        history_text = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-6:]:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                history_parts.append(f"{role}: {msg.get('content', '')}")
            history_text = "\n".join(history_parts) + "\n\n"
        
        prompt = f"""Based on the following context and conversation history, please answer the user's question. If the context doesn't contain relevant information, say so politely.

Context:
{context}

{history_text}User: {question}

Assistant: """
        
        return prompt