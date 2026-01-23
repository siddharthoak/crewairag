import os
from pathlib import Path
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class KnowledgeBaseIngestion:
    def __init__(
        self,
        kb_folder: str = "kb",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text"
    ):
        self.kb_folder = Path(__file__).parent.parent.parent / kb_folder
        self.collection_name = collection_name
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        ollama_model = os.getenv("EMBEDDING_MODEL", ollama_model)
        # Use Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_model
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_documents(self) -> List:
        """Load all documents from the kb folder."""
        documents = []
        
        for file_path in self.kb_folder.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix in [".txt", ".md"]:
                        loader = TextLoader(str(file_path))
                    else:
                        continue
                    
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def create_collection(self, vector_size: int = 768):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def ingest(self):
        """Main ingestion pipeline."""
        print("Starting ingestion process...")
        
        # Load documents
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        
        # Create embeddings
        texts = [doc.page_content for doc in splits]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Create collection
        self.create_collection(vector_size=len(embeddings[0]))
        
        # Prepare points for Qdrant
        points = []
        for i, (text, embedding, doc) in enumerate(zip(texts, embeddings, splits)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "source": doc.metadata.get("source", "unknown"),
                    "metadata": doc.metadata
                }
            )
            points.append(point)
        
        # Upload to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
        
        print(f"Successfully ingested {len(points)} chunks into Qdrant")


if __name__ == "__main__":
    ingestion = KnowledgeBaseIngestion()
    ingestion.ingest()
