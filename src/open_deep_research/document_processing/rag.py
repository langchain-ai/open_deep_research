# src/open_deep_research/document_processing/rag.py

from typing import Dict, List, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .vector_store import FAISSVectorStore

class HybridRetrievalSystem:
    """Hybrid retrieval system combining vector search and web search"""
    
    def __init__(
        self, 
        vector_store: FAISSVectorStore,
        reranker_model: str = "gpt-3.5-turbo",
        max_results: int = 10
    ):
        """Initialize hybrid retrieval system"""
        self.vector_store = vector_store
        self.reranker = ChatOpenAI(model=reranker_model)
        self.max_results = max_results
        
    async def search(
        self, 
        query: str, 
        web_results: List[Document] = None,
        k_local: int = 5,
        include_sources: List[str] = None
    ) -> List[Document]:
        """Search for relevant documents using hybrid approach"""
        results = []
        
        # Get local documents from vector store
        local_results = self.vector_store.search(query, k=k_local)
        
        # Filter by source if specified
        if include_sources:
            local_results = [
                doc for doc in local_results 
                if doc.metadata.get("source") in include_sources
            ]
            
        # Combine with web results if available
        if web_results:
            results = local_results + web_results
        else:
            results = local_results
            
        # Apply reranking if we have enough results
        if len(results) > 1:
            results = await self._rerank_results(query, results)
            
        # Return top results
        return results[:self.max_results]
        
    async def _rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank results using LLM-based reranking"""
        if not documents:
            return []
            
        # Create context for reranking
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}..." 
            for i, doc in enumerate(documents)
        ])
        
        # Use LLM to rerank results
        prompt = f"""Given the query: "{query}"
        
        And the following documents:
        
        {context}
        
        Rank the documents by relevance to the query. Return a comma-separated list of document numbers from most to least relevant.
        For example: "3,1,5,2,4"
        
        Document rankings:"""
        
        # Get reranking from LLM
        response = await self.reranker.ainvoke(prompt)
        rankings_text = response.content.strip()
        
        # Parse rankings
        try:
            rankings = [int(x.strip()) - 1 for x in rankings_text.split(",")]
            
            # Reorder documents based on rankings
            reranked = []
            for idx in rankings:
                if 0 <= idx < len(documents):
                    reranked.append(documents[idx])
                    
            # Add any documents not mentioned in rankings
            mentioned = set(rankings)
            for i, doc in enumerate(documents):
                if i not in mentioned and 0 <= i < len(documents):
                    reranked.append(doc)
                    
            return reranked
        except:
            # Fallback if parsing fails
            return documents