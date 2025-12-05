"""
ChromaDB Manager Module

Manages ChromaDB setup and operations.
"""

from pathlib import Path
from typing import Optional, List, Dict
import warnings

# Suppress ChromaDB warnings about Python 3.14 compatibility
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

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
from ..embeddings.model_registry import get_model_registry


class ChromaDBManager:
    """Manages ChromaDB database and collections."""

    def __init__(
        self, db_path: Optional[str] = None, collection_name: Optional[str] = None
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

    def _generate_collection_name(
        self, model_name: Optional[str] = None, base_name: Optional[str] = None
    ) -> str:
        """
        Generate a model-specific collection name.

        Args:
            model_name: Name of the embedding model (uses config default if None)
            base_name: Base collection name (uses config default if None)

        Returns:
            Model-specific collection name like "document_embeddings_bge_large"
        """
        # Get model name from config if not provided
        if model_name is None:
            config = get_config()
            model_name = config.MODEL_NAME

        # Get base name from config if not provided
        if base_name is None:
            base_name = self.collection_name

        # Extract model slug from model name
        # Examples: "BAAI/bge-large-en-v1.5" -> "bge_large", "google/embeddinggemma-300m" -> "gemma_300m"
        model_slug = self._extract_model_slug(model_name)

        # Create collection name: base_name + model_slug
        if model_slug:
            collection_name = f"{base_name}_{model_slug}"
        else:
            # Fallback to base name if slug extraction fails
            collection_name = base_name

        # Ensure collection name is valid for ChromaDB (alphanumeric, underscore, dash)
        collection_name = self._sanitize_collection_name(collection_name)

        return collection_name

    def _extract_model_slug(self, model_name: str) -> str:
        """
        Extract a short, filesystem-safe slug from model name.

        Args:
            model_name: Full model name (e.g., "BAAI/bge-large-en-v1.5")

        Returns:
            Short slug (e.g., "bge_large")
        """
        import re

        # Common patterns to extract meaningful parts
        # Handle HuggingFace model names like "org/model-name-version"
        if "/" in model_name:
            # Take the part after the last "/"
            model_part = model_name.split("/")[-1]
        else:
            model_part = model_name

        # Remove version numbers and common suffixes
        model_part = re.sub(
            r"[-_]v?\d+(\.\d+)*.*$", "", model_part
        )  # Remove version suffixes
        model_part = re.sub(r"[-_]en.*$", "", model_part)  # Remove language suffixes

        # Extract key components (take first 2-3 meaningful words)
        words = re.findall(r"[a-zA-Z]+", model_part)
        if len(words) >= 2:
            # Take first 2 words, or first word + important keywords
            if "bge" in words[0].lower():
                slug = "bge_" + "_".join(words[1:2]) if len(words) > 1 else "bge"
            elif "embedding" in words[0].lower() or "gemma" in words[0].lower():
                slug = "_".join(words[:2])  # e.g., "embedding_gemma"
            else:
                slug = "_".join(words[:2])  # Take first 2 words
        else:
            # Fallback: take first meaningful word or use "unknown"
            slug = words[0] if words else "unknown"

        # Ensure it's not too long (max 20 chars for readability)
        return slug.lower()[:20]

    def _sanitize_collection_name(self, name: str) -> str:
        """
        Sanitize collection name to be valid for ChromaDB.

        Args:
            name: Raw collection name

        Returns:
            Sanitized collection name
        """
        import re

        # Replace invalid characters with underscores
        # Allow: alphanumeric, underscore, dash
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Remove multiple consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure not empty
        if not sanitized:
            sanitized = "default_collection"

        return sanitized

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
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
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
        embedding_dimension: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> chromadb.Collection:
        """
        Get or create a collection.

        Args:
            name: Collection name (uses model-specific name if None and model_name provided)
            embedding_dimension: Embedding dimension (retrieved from model metadata if None)
            model_name: Embedding model name (uses config default if None)

        Returns:
            ChromaDB collection instance
        """
        # Determine collection name
        if name is None and model_name is not None:
            # Generate model-specific collection name
            collection_name = self._generate_collection_name(model_name)
        else:
            collection_name = name or self.collection_name

        # Determine embedding dimension
        if embedding_dimension is None:
            if model_name is not None:
                # Get dimension from model metadata
                try:
                    registry = get_model_registry()
                    metadata = registry.get_model_metadata(model_name)
                    if metadata:
                        embedding_dimension = metadata.embedding_dimension
                    else:
                        # Fallback to config
                        config = get_config()
                        embedding_dimension = config.get_embedding_dimension()
                except Exception as e:
                    self.logger.warning(
                        f"Could not get embedding dimension from model metadata: {e}"
                    )
                    config = get_config()
                    embedding_dimension = config.get_embedding_dimension()
            else:
                # Use config default
                config = get_config()
                embedding_dimension = config.get_embedding_dimension()

        if self.collection is not None and self.collection.name == collection_name:
            return self.collection

        client = self.get_client()

        try:
            # Try to get existing collection
            self.collection = client.get_collection(name=collection_name)
            self.logger.info(f"Retrieved existing collection: {collection_name}")

            # Validate collection metadata against expected parameters
            self._validate_collection_metadata(
                collection=self.collection,
                expected_dimension=embedding_dimension,
                expected_model=model_name,
            )

        except Exception:
            # Collection doesn't exist, create it
            try:
                self.logger.info(f"Creating new collection: {collection_name}")

                # Create collection with model metadata
                metadata = {
                    "embedding_dimension": embedding_dimension,
                    "created_at": str(self._get_current_timestamp()),
                }

                # Add model information if available
                if model_name is not None:
                    metadata["model_name"] = model_name
                    try:
                        registry = get_model_registry()
                        model_metadata = registry.get_model_metadata(model_name)
                        if model_metadata:
                            metadata["model_adapter"] = (
                                model_metadata.adapter_class.__name__
                            )
                            metadata["max_sequence_length"] = (
                                model_metadata.max_sequence_length
                            )
                            metadata["precision_requirements"] = str(
                                model_metadata.precision_requirements
                            )
                    except Exception as e:
                        self.logger.debug(f"Could not add model metadata: {e}")

                self.collection = client.create_collection(
                    name=collection_name, metadata=metadata
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
            self.logger.debug(
                f"ChromaDB health check passed: {len(collections)} collections"
            )
            return True
        except Exception as e:
            self.logger.error(f"ChromaDB health check failed: {e}")
            return False

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def list_collections_by_model(self) -> Dict[str, str]:
        """
        List all collections grouped by model.

        Returns:
            Dictionary mapping model names to collection names.
            Collections without model metadata are grouped under "unknown".
        """
        collections = self.list_collections()
        model_collections = {}

        for collection_name in collections:
            try:
                # Get collection metadata to determine model
                collection = self.get_client().get_collection(name=collection_name)
                metadata = collection.metadata or {}

                model_name = metadata.get("model_name", "unknown")
                model_collections[model_name] = collection_name

            except Exception as e:
                self.logger.debug(
                    f"Could not get metadata for collection {collection_name}: {e}"
                )
                model_collections.setdefault("unknown", []).append(collection_name)

        return model_collections
    
    def auto_detect_model(self) -> Optional[str]:
        """
        Auto-detect the embedding model from available collections.
        
        Returns the model name from the first non-empty collection found.
        Prioritizes collections with more documents.
        
        Returns:
            Model name if found, None otherwise
        """
        try:
            collections = self.list_collections()
            if not collections:
                self.logger.debug("No collections found for auto-detection")
                return None
            
            # Get collection stats and sort by count
            collection_stats = []
            for collection_name in collections:
                try:
                    collection = self.get_client().get_collection(name=collection_name)
                    count = collection.count()
                    metadata = collection.metadata or {}
                    model_name = metadata.get("model_name")
                    
                    if model_name and count > 0:
                        collection_stats.append({
                            "name": collection_name,
                            "model": model_name,
                            "count": count
                        })
                except Exception as e:
                    self.logger.debug(f"Could not check collection {collection_name}: {e}")
                    continue
            
            if not collection_stats:
                self.logger.debug("No non-empty collections with model metadata found")
                return None
            
            # Sort by count (descending) and get the model from the largest collection
            collection_stats.sort(key=lambda x: x["count"], reverse=True)
            detected_model = collection_stats[0]["model"]
            
            self.logger.info(
                f"Auto-detected model '{detected_model}' from collection '{collection_stats[0]['name']}' "
                f"({collection_stats[0]['count']} documents)"
            )
            
            return detected_model
            
        except Exception as e:
            self.logger.error(f"Failed to auto-detect model: {e}")
            return None

    def get_collection_for_model(self, model_name: str) -> chromadb.Collection:
        """
        Get the collection associated with a specific model.

        Args:
            model_name: Name of the embedding model

        Returns:
            ChromaDB collection for the specified model
        """
        collection_name = self._generate_collection_name(model_name)
        return self.get_or_create_collection(
            name=collection_name, model_name=model_name
        )

    def validate_collection_model(
        self, collection_name: str, expected_model_name: str
    ) -> Dict[str, any]:
        """
        Validate that a collection matches the expected model.

        Args:
            collection_name: Name of the collection to validate
            expected_model_name: Expected model name

        Returns:
            Validation result with 'valid' boolean and error details if invalid
        """
        try:
            collection = self.get_client().get_collection(name=collection_name)
            metadata = collection.metadata or {}

            stored_model = metadata.get("model_name")
            stored_dimension = metadata.get("embedding_dimension")

            # Check model name
            model_valid = stored_model == expected_model_name

            # Check embedding dimension
            try:
                registry = get_model_registry()
                expected_metadata = registry.get_model_metadata(expected_model_name)
                expected_dimension = (
                    expected_metadata.embedding_dimension if expected_metadata else None
                )
                dimension_valid = stored_dimension == expected_dimension
            except Exception:
                dimension_valid = True  # Skip dimension check if registry unavailable

            is_valid = model_valid and dimension_valid

            result = {
                "valid": is_valid,
                "collection_name": collection_name,
                "expected_model": expected_model_name,
                "stored_model": stored_model,
                "model_valid": model_valid,
                "dimension_valid": dimension_valid,
            }

            if not is_valid:
                errors = []
                if not model_valid:
                    errors.append(
                        f"Model mismatch: expected '{expected_model_name}', found '{stored_model}'"
                    )
                if not dimension_valid:
                    errors.append(
                        f"Dimension mismatch: expected {expected_dimension}, found {stored_dimension}"
                    )
                result["errors"] = errors

            return result

        except Exception as e:
            return {
                "valid": False,
                "collection_name": collection_name,
                "expected_model": expected_model_name,
                "error": str(e),
            }

    def migrate_collection(
        self, old_model_name: str, new_model_name: str, delete_old: bool = False
    ) -> Dict[str, any]:
        """
        Migrate data from one model collection to another.

        NOTE: This is a stub implementation. Full migration requires:
        1. Re-embedding all documents with the new model
        2. Creating new collection with correct metadata
        3. Transferring re-embedded data
        4. Optionally deleting old collection

        Args:
            old_model_name: Source model name
            new_model_name: Target model name
            delete_old: Whether to delete the old collection after migration

        Returns:
            Migration result (currently just a placeholder)
        """
        self.logger.warning(
            "Collection migration is not implemented. "
            "Migration requires re-embedding all documents with the new model."
        )

        return {
            "migrated": False,
            "old_model": old_model_name,
            "new_model": new_model_name,
            "reason": "Re-embedding required - not implemented in this phase",
            "delete_old": delete_old,
        }

    def _validate_collection_metadata(
        self,
        collection: chromadb.Collection,
        expected_dimension: int,
        expected_model: Optional[str] = None,
    ) -> None:
        """
        Validate collection metadata against expected parameters.

        Args:
            collection: ChromaDB collection to validate
            expected_dimension: Expected embedding dimension
            expected_model: Expected model name (optional)

        Raises:
            ValueError: If collection metadata doesn't match expectations
        """
        metadata = collection.metadata or {}

        # Check embedding dimension
        collection_dimension = metadata.get("embedding_dimension")
        if collection_dimension is not None:
            try:
                collection_dimension = int(collection_dimension)
                if collection_dimension != expected_dimension:
                    raise ValueError(
                        f"Collection '{collection.name}' has embedding dimension {collection_dimension}, "
                        f"but expected {expected_dimension}. "
                        f"This suggests the collection was created with a different model. "
                        f"Please use a different collection name or delete the existing collection."
                    )
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f"Could not parse embedding dimension from collection '{collection.name}': {e}"
                )

        # Check model compatibility (warning only, not error)
        collection_model = metadata.get("model_name")
        if expected_model and collection_model and collection_model != expected_model:
            # Check if models are compatible (same adapter type)
            try:
                registry = get_model_registry()
                expected_type = registry.detect_model_type(expected_model)
                collection_type = registry.detect_model_type(collection_model)

                if expected_type != collection_type:
                    self.logger.warning(
                        f"Collection '{collection.name}' was created with {collection_model} "
                        f"(type: {collection_type}), but you're using {expected_model} "
                        f"(type: {expected_type}). "
                        f"Consider using model-specific collection names to avoid confusion."
                    )
                else:
                    self.logger.info(
                        f"Collection '{collection.name}' compatible: "
                        f"{collection_model} → {expected_model}"
                    )
            except Exception as e:
                self.logger.debug(f"Could not check model compatibility: {e}")

    def validate_collection_model(
        self, collection_name: str, expected_model_name: str
    ) -> Dict[str, any]:
        """
        Validate that a collection matches an expected model.

        Args:
            collection_name: Name of the collection to validate
            expected_model_name: Expected model name

        Returns:
            Validation result dictionary
        """
        try:
            client = self.get_client()
            collection = client.get_collection(name=collection_name)
            metadata = collection.metadata or {}

            collection_model = metadata.get("model_name")
            collection_dimension = metadata.get("embedding_dimension")

            # Get expected metadata
            registry = get_model_registry()
            expected_metadata = registry.get_model_metadata(expected_model_name)

            result = {
                "collection_name": collection_name,
                "expected_model": expected_model_name,
                "collection_model": collection_model,
                "collection_dimension": collection_dimension,
                "valid": True,
                "warnings": [],
                "errors": [],
            }

            if collection_model and collection_model != expected_model_name:
                result["warnings"].append(
                    f"Collection model mismatch: {collection_model} vs {expected_model_name}"
                )

            if expected_metadata and collection_dimension:
                try:
                    expected_dim = expected_metadata.embedding_dimension
                    collection_dim = int(collection_dimension)
                    if collection_dim != expected_dim:
                        result["errors"].append(
                            f"Dimension mismatch: {collection_dim} vs {expected_dim}"
                        )
                        result["valid"] = False
                except (ValueError, TypeError):
                    result["warnings"].append("Could not parse collection dimension")

            return result

        except Exception as e:
            return {
                "collection_name": collection_name,
                "expected_model": expected_model_name,
                "valid": False,
                "errors": [f"Collection access failed: {str(e)}"],
                "warnings": [],
            }
