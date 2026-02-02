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
        
        # Better separators for academic papers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                "! ",
                "? ",
                "; ",
                ": ",
                " ",     # Words
                ""       # Characters
            ],
            length_function=len,
            is_separator_regex=False,
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
                    # Add source info to each doc
                    for doc in docs:
                        doc.metadata['source'] = str(file_path)
                    documents.extend(docs)
                    print(f"✅ Loaded: {file_path.name} ({len(docs)} pages)")
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
        
        return documents
    
    def create_collection(self, vector_size: int = 768, recreate: bool = False):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Option to recreate collection (useful for re-ingestion)
        if recreate and self.collection_name in collection_names:
            print(f"🗑️  Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(collection_name=self.collection_name)
            collection_names.remove(self.collection_name)
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {self.collection_name}")
        else:
            print(f"ℹ️  Collection {self.collection_name} already exists")
    
    def ingest(self, recreate: bool = False):
        """Main ingestion pipeline."""
        print("\n" + "="*80)
        print("🚀 STARTING INGESTION PROCESS")
        print("="*80 + "\n")
        
        # Load documents
        print("📂 Loading documents...")
        documents = self.load_documents()
        print(f"✅ Loaded {len(documents)} document pages\n")
        
        if not documents:
            print("❌ No documents found! Check your kb folder.")
            return
        
        # Split documents
        print("✂️  Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(splits)} chunks\n")
        
        # Show sample chunks for verification
        print("📋 Sample chunks:")
        for i, split in enumerate(splits[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Source: {split.metadata.get('source', 'unknown')}")
            print(f"Page: {split.metadata.get('page', 'N/A')}")
            print(f"Content preview: {split.page_content[:200]}...")
        print()
        
        # Create embeddings
        print("🔢 Generating embeddings...")
        texts = [doc.page_content for doc in splits]
        embeddings = self.embeddings.embed_documents(texts)
        print(f"✅ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})\n")
        
        # Create collection
        print("📦 Setting up Qdrant collection...")
        self.create_collection(vector_size=len(embeddings[0]), recreate=recreate)
        print()
        
        # Prepare points for Qdrant
        print("🔨 Preparing data points...")
        points = []
        for i, (text, embedding, doc) in enumerate(zip(texts, embeddings, splits)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", None),
                    "metadata": doc.metadata
                }
            )
            points.append(point)
        print(f"✅ Prepared {len(points)} points\n")
        
        # Upload to Qdrant in batches
        print("⬆️  Uploading to Qdrant...")
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            batch_num = i // batch_size + 1
            print(f"  ✅ Batch {batch_num}/{total_batches} uploaded ({len(batch)} points)")
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS! Ingested {len(points)} chunks into Qdrant")
        print(f"{'='*80}\n")
        
        # Verify ingestion
        self.verify_ingestion()
    
    def verify_ingestion(self):
        """Verify what was ingested by running test searches."""
        print("\n🔍 VERIFICATION - Running test searches...\n")
        
        test_queries = [
            "DSPY",
            "compiler",
            "machine learning",
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            query_vector = self.embeddings.embed_query(query)
            
            # ✅ Use query_points instead of search
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=3
            ).points
            
            if results:
                print(f"  ✅ Found {len(results)} results")
                print(f"  Top result score: {results[0].score:.4f}")
                print(f"  Preview: {results[0].payload['text'][:100]}...")
            else:
                print(f"  ❌ No results found")
            print()


if __name__ == "__main__":
    # Add option to recreate collection
    import sys
    recreate = "--recreate" in sys.argv
    
    if recreate:
        print("\n⚠️  WARNING: Will recreate collection (delete existing data)")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    ingestion = KnowledgeBaseIngestion()
    ingestion.ingest(recreate=recreate)