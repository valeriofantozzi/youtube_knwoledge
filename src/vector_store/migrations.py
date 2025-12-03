"""
Schema Migration Utilities

Provides utilities for migrating metadata schemas in ChromaDB.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from .chroma_manager import ChromaDBManager
from .schema import ChunkMetadata, chromadb_metadata_to_schema
from ..utils.logger import get_default_logger
from ..utils.config import get_config


class SchemaMigrator:
    """Handles schema migrations for vector store metadata."""
    
    def __init__(self, chroma_manager: Optional[ChromaDBManager] = None):
        """
        Initialize schema migrator.
        
        Args:
            chroma_manager: ChromaDBManager instance (creates new if None)
        """
        self.chroma_manager = chroma_manager or ChromaDBManager()
        self.config = get_config()
        self.logger = get_default_logger()
        self.migrations_dir = Path(self.config.get("MIGRATIONS_DIR", "./data/migrations"))
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
    
    def get_current_schema_version(self) -> str:
        """
        Get current schema version.
        
        Returns:
            Schema version string
        """
        # Current schema version
        return "1.0.0"
    
    def create_migration(
        self,
        migration_name: str,
        description: str,
        up_migration: callable,
        down_migration: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Create a new migration.
        
        Args:
            migration_name: Name of migration
            description: Description of migration
            up_migration: Function to apply migration
            down_migration: Optional function to rollback migration
        
        Returns:
            Migration metadata dictionary
        """
        migration = {
            "name": migration_name,
            "description": description,
            "version": self.get_current_schema_version(),
            "created_at": datetime.now().isoformat(),
            "up": up_migration,
            "down": down_migration
        }
        
        # Save migration metadata
        migration_file = self.migrations_dir / f"{migration_name}.json"
        migration_metadata = {
            "name": migration_name,
            "description": description,
            "version": migration["version"],
            "created_at": migration["created_at"]
        }
        
        with open(migration_file, 'w') as f:
            json.dump(migration_metadata, f, indent=2)
        
        self.logger.info(f"Created migration: {migration_name}")
        return migration
    
    def list_migrations(self) -> List[Dict[str, Any]]:
        """
        List all available migrations.
        
        Returns:
            List of migration metadata dictionaries
        """
        migrations = []
        
        for migration_file in self.migrations_dir.glob("*.json"):
            try:
                with open(migration_file, 'r') as f:
                    migration_data = json.load(f)
                    migrations.append(migration_data)
            except Exception as e:
                self.logger.warning(f"Failed to load migration {migration_file}: {e}")
        
        return sorted(migrations, key=lambda x: x.get("created_at", ""))
    
    def validate_schema(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata against current schema.
        
        Args:
            metadata: Metadata dictionary to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            chunk_metadata = chromadb_metadata_to_schema(metadata)
            return chunk_metadata.validate()
        except Exception as e:
            self.logger.debug(f"Schema validation failed: {e}")
            return False
    
    def migrate_metadata(
        self,
        old_metadata: Dict[str, Any],
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Migrate metadata to target schema version.
        
        Args:
            old_metadata: Old metadata dictionary
            target_version: Target schema version (default: current)
        
        Returns:
            Migrated metadata dictionary
        """
        target_version = target_version or self.get_current_schema_version()
        
        # For now, just validate and return
        # In the future, this could apply actual migrations
        if self.validate_schema(old_metadata):
            return old_metadata
        
        # If validation fails, try to fix common issues
        migrated = old_metadata.copy()
        
        # Ensure all required fields exist
        required_fields = ["source_id", "date", "title", "chunk_index", "chunk_id", "token_count", "filename"]
        for field in required_fields:
            if field not in migrated:
                migrated[field] = "" if isinstance(migrated.get(field, ""), str) else 0
        
        return migrated
    
    def backup_collection_schema(
        self,
        collection_name: Optional[str] = None,
        backup_file: Optional[Path] = None
    ) -> Path:
        """
        Backup collection schema and metadata.
        
        Args:
            collection_name: Collection name (default from config)
            backup_file: Backup file path (auto-generated if None)
        
        Returns:
            Path to backup file
        """
        collection_name = collection_name or self.chroma_manager.collection_name
        
        if backup_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.migrations_dir / f"schema_backup_{collection_name}_{timestamp}.json"
        
        collection = self.chroma_manager.get_or_create_collection(collection_name)
        
        # Get sample metadata to understand schema
        try:
            sample = collection.get(limit=1)
            schema_info = {
                "collection_name": collection_name,
                "schema_version": self.get_current_schema_version(),
                "backup_date": datetime.now().isoformat(),
                "sample_metadata": sample.get("metadatas", [])[:1] if sample.get("metadatas") else [],
                "total_documents": collection.count()
            }
            
            with open(backup_file, 'w') as f:
                json.dump(schema_info, f, indent=2)
            
            self.logger.info(f"Backed up schema to {backup_file}")
            return backup_file
        
        except Exception as e:
            self.logger.error(f"Failed to backup schema: {e}")
            raise

