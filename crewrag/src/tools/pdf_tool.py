
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from crewai.tools import BaseTool
from langchain_ollama import OllamaEmbeddings


# Load environment variables from .env file
load_dotenv()


class QdrantSearchTool(BaseTool):
    """Tool for searching documents in Qdrant vector database."""
    name: str = "Qdrant Search Tool"
    description: str = (
        "Searches the PDF knowledge base using semantic similarity. "
        "Use this tool FIRST for any queries about DSPY, machine learning, "
        "AI frameworks, or technical documentation. "
        "Input should be a natural language question or search query. "
        "Returns the most relevant document excerpts with similarity scores."
    )
    
    client: Any = None
    embeddings: Any = None
    
    def __init__(self, host: str = "localhost", port: int = 6333, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize Qdrant client and embeddings.
        
        Args:
            host: Qdrant host (default: localhost)
            port: Qdrant port (default: 6333)
            ollama_base_url: Ollama API base URL (default: http://localhost:11434)
        """
        super().__init__()
        self.client = QdrantClient(host=host, port=port)
        
        # Initialize Ollama embeddings to match ingestion
        ollama_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_model
        )
    
    def _run(self, query_text: str, collection_name: str = "knowledge_base", limit: int = 5) -> str:
        """
        Execute the search tool.
        
        Args:
            query_text: Query string to search for
            collection_name: Name of the Qdrant collection
            limit: Maximum number of results to return
        
        Returns:
            Formatted string with search results
        """
        query_vector = self.embeddings.embed_query(query_text)
        results = self.search_documents(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. Score: {result['score']:.4f}\n"
                f"   Content: {result['payload'].get('text', 'N/A')[:200]}...\n"
                f"   Source: {result['payload'].get('source', 'N/A')}"
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No results found."
    
    def search_documents(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Qdrant.
        
        Args:
            collection_name: Name of the Qdrant collection
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply (e.g., {"source": "doc.pdf"})
        
        Returns:
            List of search results with scores and payloads
        """
        search_params = SearchParams(
            exact=False,
            quantization=None
        ) if score_threshold else None
        
        filter_obj = None
        if filters:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filters.items()
            ]
            filter_obj = Filter(must=conditions)
        
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_obj,
            search_params=search_params
        ).points
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]
    
    def get_document(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            collection_name: Name of the Qdrant collection
            document_id: ID of the document to retrieve
        
        Returns:
            Document payload or None if not found
        """
        result = self.client.retrieve(
            collection_name=collection_name,
            ids=[document_id]
        )
        
        return result[0].payload if result else None
    
    def test_search(self, collection_name: str, query_text: str, limit: int = 5):
        """
        Test search on ingested PDF documents using a query string.
        
        Args:
            collection_name: Name of the Qdrant collection
            query_text: Query string to search for
            limit: Maximum number of results to return
        
        Returns:
            List of search results with scores and content
        """
        
        # Use Ollama embeddings (same as ingestion)
        query_vector = self.embeddings.embed_query(query_text)
        
        results = self.search_documents(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        print(f"\nQuery: {query_text}")
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Content: {result['payload'].get('text', 'N/A')[:200]}...")
            print(f"   Source: {result['payload'].get('source', 'N/A')}")
            print(f"   Metadata: {result['payload'].get('metadata', {})}\n")
        
        return results


if __name__ == "__main__":
    # Example usage
    tool = QdrantSearchTool(host="localhost", port=6333)
    
    # Test with a sample query
    collection_name = "knowledge_base"  # Replace with your collection name
    query = "What is dspy?"  # Replace with your query
    
    tool.test_search(collection_name=collection_name, query_text=query, limit=5)
    
    
