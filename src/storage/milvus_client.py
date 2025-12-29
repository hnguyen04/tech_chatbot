"""
Milvus/Zilliz Cloud client for vector storage and retrieval
"""
import os
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from langchain_core.documents import Document

load_dotenv()


class MilvusRetriever:
    """
    Milvus-based retriever for Zilliz Cloud
    """
    
    def __init__(
        self,
        collection_name: str = None,
        dimension: int = None,
        uri: str = None,
        token: str = None
    ):
        """
        Initialize Milvus retriever
        
        Args:
            collection_name: Name of the Milvus collection
            dimension: Dimension of embeddings
            uri: Milvus/Zilliz Cloud URI
            token: API token for Zilliz Cloud
        """
        # Get config from environment or parameters
        self.uri = uri or os.getenv("ZILLIZ_CLOUD_URI") or os.getenv("ZILLIZIO_HOST")
        self.token = token or os.getenv("ZILLIZ_CLOUD_TOKEN") or os.getenv("ZILLIZIO_API_KEY")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION_NAME", "tech_chatbot_collection")
        self.dimension = dimension or int(os.getenv("MILVUS_EMBEDDING_DIM", "896"))
        
        if not self.uri or not self.token:
            raise ValueError(
                "Missing Zilliz Cloud credentials. Please set ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN "
                "(or ZILLIZIO_HOST and ZILLIZIO_API_KEY) in your .env file"
            )
        
        # Connect to Zilliz Cloud
        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
            print(f"✓ Connected successfully to Zilliz Cloud at {self.uri}")
            self._ensure_collection()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Zilliz Cloud: {e}")
    
    def _ensure_collection(self):
        """
        Ensure collection exists with proper schema
        """
        # Check if collection exists
        collections = self.client.list_collections()
        
        if self.collection_name in collections:
            print(f"✓ Collection '{self.collection_name}' already exists")
            return
        
        # Create collection with schema
        print(f"Creating collection '{self.collection_name}' with dimension {self.dimension}...")
        
        # Create collection with auto-generated ID and vector field
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True  # Allow dynamic metadata fields
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        
        # Create index for vector field
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",  # Auto-select best index
            metric_type="COSINE"  # or "L2", "IP"
        )
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        
        print(f"✓ Collection '{self.collection_name}' created successfully")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Add documents with their embeddings to Milvus
        
        Args:
            documents: List of LangChain Document objects
            embeddings: List of embedding vectors
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        if not documents:
            print("No documents to add")
            return
        
        # Prepare data for insertion
        data = []
        for doc, embedding in zip(documents, embeddings):
            # Serialize metadata to JSON string
            metadata_dict = doc.metadata.copy()
            
            # Ensure metadata values are JSON serializable
            for key, value in metadata_dict.items():
                if isinstance(value, (list, dict)):
                    metadata_dict[key] = json.dumps(value)
                elif value is None:
                    metadata_dict[key] = ""
                else:
                    metadata_dict[key] = str(value)
            
            data.append({
                "vector": embedding,
                "text": doc.page_content,
                "metadata": metadata_dict
            })
        
        # Insert data in batches
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )
            total_inserted += len(batch)
            print(f"Inserted batch {i//batch_size + 1}: {len(batch)} documents (Total: {total_inserted}/{len(data)})")
        
        print(f"✓ Successfully added {len(documents)} documents to Milvus")
    
    def retrieve(
        self,
        query_embedding: List[float],
        k: int = 20,
        filter_expr: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve top-k documents for a query embedding
        
        Args:
            query_embedding: Query embedding vector
            k: Number of documents to retrieve
            filter_expr: Optional filter expression (e.g., 'category == "laptop"')
            
        Returns:
            List of retrieved Document objects
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {}
        }
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="vector",
            search_params=search_params,
            limit=k,
            output_fields=["text", "metadata"],
            filter=filter_expr
        )
        
        # Convert results to LangChain Documents
        documents = []
        for hits in results:
            for hit in hits:
                metadata = hit["entity"]["metadata"]
                
                # Parse JSON strings back to objects if needed
                parsed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, str):
                        try:
                            parsed_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            parsed_metadata[key] = value
                    else:
                        parsed_metadata[key] = value
                
                doc = Document(
                    page_content=hit["entity"]["text"],
                    metadata=parsed_metadata
                )
                documents.append(doc)
        
        return documents
    
    def retrieve_with_scores(
        self,
        query_embedding: List[float],
        k: int = 20,
        filter_expr: Optional[str] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve top-k documents with similarity scores
        
        Args:
            query_embedding: Query embedding vector
            k: Number of documents to retrieve
            filter_expr: Optional filter expression
            
        Returns:
            List of tuples (Document, score)
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {}
        }
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="vector",
            search_params=search_params,
            limit=k,
            output_fields=["text", "metadata"],
            filter=filter_expr
        )
        
        # Convert results to LangChain Documents with scores
        documents_with_scores = []
        for hits in results:
            for hit in hits:
                metadata = hit["entity"]["metadata"]
                
                # Parse JSON strings back to objects if needed
                parsed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, str):
                        try:
                            parsed_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            parsed_metadata[key] = value
                    else:
                        parsed_metadata[key] = value
                
                doc = Document(
                    page_content=hit["entity"]["text"],
                    metadata=parsed_metadata
                )
                # Distance is returned; convert to similarity score if needed
                score = hit["distance"]  # For COSINE, higher is better
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    def delete_collection(self):
        """
        Delete the collection (use with caution!)
        """
        if self.collection_name in self.client.list_collections():
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"✓ Deleted collection '{self.collection_name}'")
        else:
            print(f"Collection '{self.collection_name}' does not exist")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return stats
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def collection_exists(self) -> bool:
        """
        Check if collection exists
        
        Returns:
            True if collection exists, False otherwise
        """
        return self.collection_name in self.client.list_collections()


# Legacy class for backward compatibility
class ZillizMilvusClient:
    """
    Legacy client class - use MilvusRetriever instead
    """
    
    def __init__(self):
        load_dotenv()
        host = os.getenv("ZILLIZIO_HOST") or os.getenv("ZILLIZ_CLOUD_URI")
        api_key = os.getenv("ZILLIZIO_API_KEY") or os.getenv("ZILLIZ_CLOUD_TOKEN")

        if not host or not api_key:
            raise ValueError("Missing environment variables: ZILLIZIO_HOST/ZILLIZ_CLOUD_URI or ZILLIZIO_API_KEY/ZILLIZ_CLOUD_TOKEN")

        try:
            self.client = MilvusClient(uri=host, token=api_key)
            print(f"Connected successfully to Zilliz Cloud Milvus at {host}")
        except Exception as e:
            print(f"Failed to connect to Zilliz Cloud Milvus: {e}")
            self.client = None
