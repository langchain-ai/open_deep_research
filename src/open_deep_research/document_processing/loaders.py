# src/open_deep_research/document_processing/loaders.py

import os
from typing import Dict, List, Optional
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

class DocumentLoader:
    """Base class for document loaders"""
    
    def load(self, file_path: str) -> List[Document]:
        """Load document from file path"""
        raise NotImplementedError
        
    def get_text_splitter(self) -> TextSplitter:
        """Get text splitter for chunking"""
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

class PDFLoader(DocumentLoader):
    """Loader for PDF files"""
    
    def load(self, file_path: str) -> List[Document]:
        """Load PDF document and extract text with metadata"""
        if not file_path.endswith('.pdf'):
            raise ValueError(f"Not a PDF file: {file_path}")
            
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)
        
        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
                
            metadata = {
                "source": file_path,
                "file_name": file_name,
                "page": i + 1,
                "total_pages": len(reader.pages)
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
            
        text_splitter = self.get_text_splitter()
        return text_splitter.split_documents(documents)

class ExcelLoader(DocumentLoader):
    """Loader for Excel files"""
    
    def load(self, file_path: str) -> List[Document]:
        """Load Excel document and extract data with metadata"""
        if not file_path.endswith(('.xlsx', '.xls')):
            raise ValueError(f"Not an Excel file: {file_path}")
            
        file_name = os.path.basename(file_path)
        excel_file = pd.ExcelFile(file_path)
        
        documents = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Convert dataframe to text representation
            text = f"Sheet: {sheet_name}\n\n"
            text += df.to_string(index=False)
            
            metadata = {
                "source": file_path,
                "file_name": file_name,
                "sheet_name": sheet_name
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
            
        text_splitter = self.get_text_splitter()
        return text_splitter.split_documents(documents)