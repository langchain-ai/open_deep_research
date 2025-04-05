# src/open_deep_research/document_processing/vector_store.py

import os
import pickle
from typing import Dict, List, Optional, Any
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

class FAISSVectorStore:
    """FAISS vector store for document embeddings"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        """Initialize FAISS vector store"""
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.index = None
        self.documents = []
        self.document_embeddings = []
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if not documents:
            return
            
        # Get embeddings for documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.document_embeddings.extend(embeddings)
        
        # Build FAISS index
        dimension = len(embeddings[0])
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
            
        # Add embeddings to index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search for similar documents
        query_embedding_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_array)
        distances, indices = self.index.search(query_embedding_array, k)
        
        # Return relevant documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append(self.documents[idx])
                
        return results
        
    def save(self, directory: str) -> None:
        """Save vector store to disk"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
            
        # Save documents with pickle
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
    def load(self, directory: str) -> None:
        """Load vector store from disk"""
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
            
        # Load FAISS index
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
        # Load documents from pickle
        documents_path = os.path.join(directory, "documents.pkl")
        if os.path.exists(documents_path):
            with open(documents_path, "rb") as f:
                self.documents = pickle.load(f)