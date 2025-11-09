"""
ChromaDB Manager Module

Manages ChromaDB setup and operations.
"""

from pathlib import Path
from typing import Optional, List, Dict
import warnings

# Suppress ChromaDB warnings about Python 3.14 compatibility
warnings.filterwarnings('ignore', category=UserWarning, module='chromadb')

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

from ..utils.config import get_config
from ..utils.logger import get_default_logger


class ChromaDBManager:
    """Manages ChromaDB database and collections."""
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            db_path: Path to ChromaDB database (default from config)
            collection_name: Name of collection (default from config)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not available. Please install it or check compatibility."
            )
        
        self.config = get_config()
        self.db_path = Path(db_path or self.config.VECTOR_DB_PATH)
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        self.logger = get_default_logger()
        
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> chromadb.Client:
        """
        Initialize ChromaDB client.
        
        Returns:
            ChromaDB client instance
        """
        if self.client is not None:
            return self.client
        
        try:
            self.logger.info(f"Initializing ChromaDB at {self.db_path}")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.logger.info("ChromaDB client initialized successfully")
            return self.client
        
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise
    
    def get_client(self) -> chromadb.Client:
        """
        Get ChromaDB client (initializes if needed).
        
        Returns:
            ChromaDB client instance
        """
        if self.client is None:
            return self.initialize()
        return self.client
    
    def get_or_create_collection(
        self,
        name: Optional[str] = None,
        embedding_dimension: int = 1024
    ) -> chromadb.Collection:
        """
        Get or create a collection.
        
        Args:
            name: Collection name (uses default if None)
            embedding_dimension: Embedding dimension (default: 1024 for BGE-large)
        
        Returns:
            ChromaDB collection instance
        """
        collection_name = name or self.collection_name
        
        if self.collection is not None and self.collection.name == collection_name:
            return self.collection
        
        client = self.get_client()
        
        try:
            # Try to get existing collection
            self.collection = client.get_collection(name=collection_name)
            self.logger.info(f"Retrieved existing collection: {collection_name}")
        
        except Exception:
            # Collection doesn't exist, create it
            try:
                self.logger.info(f"Creating new collection: {collection_name}")
                
                # Create collection with custom embedding function
                # We'll provide embeddings directly, so we use a dummy function
                self.collection = client.create_collection(
                    name=collection_name,
                    metadata={"embedding_dimension": embedding_dimension}
                )
                
                self.logger.info(f"Collection created: {collection_name}")
            
            except Exception as e:
                self.logger.error(f"Failed to create collection: {e}", exc_info=True)
                raise
        
        return self.collection
    
    def delete_collection(self, name: Optional[str] = None) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name (uses default if None)
        """
        collection_name = name or self.collection_name
        client = self.get_client()
        
        try:
            client.delete_collection(name=collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
            
            # Reset collection reference if it was the current one
            if self.collection and self.collection.name == collection_name:
                self.collection = None
        
        except Exception as e:
            self.logger.warning(f"Failed to delete collection: {e}")
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        client = self.get_client()
        collections = client.list_collections()
        return [col.name for col in collections]
    
    def get_collection_stats(self, name: Optional[str] = None) -> Dict:
        """
        Get statistics for a collection.
        
        Args:
            name: Collection name (uses default if None)
        
        Returns:
            Dictionary with collection statistics
        """
        collection_name = name or self.collection_name
        collection = self.get_or_create_collection(name=collection_name)
        
        try:
            count = collection.count()
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata or {},
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {
                "name": collection_name,
                "count": 0,
                "error": str(e),
            }
    
    def reset_database(self) -> None:
        """Reset the entire database (delete all collections)."""
        client = self.get_client()
        
        try:
            collections = client.list_collections()
            for col in collections:
                client.delete_collection(name=col.name)
            
            self.collection = None
            self.logger.info("Database reset successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to reset database: {e}", exc_info=True)
            raise
    
    def health_check(self) -> bool:
        """
        Perform health check on ChromaDB.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            client = self.get_client()
            collections = client.list_collections()
            self.logger.debug(f"ChromaDB health check passed: {len(collections)} collections")
            return True
        except Exception as e:
            self.logger.error(f"ChromaDB health check failed: {e}")
            return False
