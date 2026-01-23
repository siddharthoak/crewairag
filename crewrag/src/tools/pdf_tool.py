import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from crewai.tools import BaseTool
from langchain_ollama import OllamaEmbeddings

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantSearchTool(BaseTool):
    """Tool for searching documents in Qdrant vector database."""
    name: str = "Search PDF Knowledge Base"
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
        logger.info(f"Initializing QdrantSearchTool with embedding model: {ollama_model}")
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_model
        )
    
    def _run(self, query_text: str) -> str:
        """
        Execute the search tool.
        
        Args:
            query_text: Query string to search for
        
        Returns:
            Formatted string with search results
        """
        try:
            # Log what we received
            logger.info(f"QdrantSearchTool received query: '{query_text}'")
            logger.info(f"Query type: {type(query_text)}")
            
            # Clean the query text
            clean_query = str(query_text).strip()
            logger.info(f"Cleaned query: '{clean_query}'")
            
            # Generate embedding
            logger.info("Generating query embedding...")
            query_vector = self.embeddings.embed_query(clean_query)
            logger.info(f"Embedding generated, vector length: {len(query_vector)}")
            
            # Search
            logger.info("Searching Qdrant...")
            results = self.search_documents(
                collection_name="knowledge_base",
                query_vector=query_vector,
                limit=5
            )
            
            logger.info(f"Found {len(results)} results")
            
            if not results:
                return "No results found in the PDF knowledge base. Try web search."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                score = result['score']
                content = result['payload'].get('text', 'N/A')[:300]  # Show more content
                source = result['payload'].get('source', 'N/A')
                
                logger.info(f"Result {i}: Score={score:.4f}, Source={source}")
                
                formatted_results.append(
                    f"Result {i} (Relevance: {score:.4f}):\n"
                    f"Content: {content}...\n"
                    f"Source: {source}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            error_msg = f"Error in QdrantSearchTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error searching knowledge base: {str(e)}"
    
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
        try:
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
            
            logger.info(f"Querying collection: {collection_name}")
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
        except Exception as e:
            logger.error(f"Error in search_documents: {e}", exc_info=True)
            raise
    
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