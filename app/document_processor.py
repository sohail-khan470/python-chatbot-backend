from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document
import aiofiles
import os
from typing import List
import tempfile

class DocumentProcessor:
    @staticmethod
    async def process_uploaded_file(file_path: str, filename: str) -> List[Document]:
        """Process uploaded file based on its type"""
        
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return await DocumentProcessor._process_pdf(file_path, filename)
        elif file_extension == 'txt':
            return await DocumentProcessor._process_text(file_path, filename)
        elif file_extension == 'csv':
            return await DocumentProcessor._process_csv(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    async def _process_pdf(file_path: str, filename: str) -> List[Document]:
        """Process PDF file"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata['source'] = filename
            doc.metadata['type'] = 'pdf'
        
        return documents
    
    @staticmethod
    async def _process_text(file_path: str, filename: str) -> List[Document]:
        """Process text file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        document = Document(
            page_content=content,
            metadata={'source': filename, 'type': 'text'}
        )

        return [document]

    @staticmethod
    async def _process_csv(file_path: str, filename: str) -> List[Document]:
        """Process CSV file"""
        loader = CSVLoader(file_path)
        documents = loader.load()

        # Add source metadata
        for doc in documents:
            doc.metadata['source'] = filename
            doc.metadata['type'] = 'csv'

        return documents