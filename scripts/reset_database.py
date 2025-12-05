#!/usr/bin/env python3
"""Reset the vector database (clear all data)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.pipeline import VectorStorePipeline


def reset_database():
    """Delete all documents from the collection."""
    print("⚠️  WARNING: This will delete ALL documents from the vector database!")
    print("Make sure you have a backup before proceeding.")
    
    confirm = input("\nType 'yes' to confirm deletion: ")
    
    if confirm.lower() != 'yes':
        print("Aborted.")
        return False
    
    try:
        pipeline = VectorStorePipeline()
        pipeline.chroma_manager.initialize()
        client = pipeline.chroma_manager.get_client()
        
        collection = pipeline.chroma_manager.get_or_create_collection()
        
        count_before = collection.count()
        print(f"\n🗑️  Deleting {count_before} documents...")
        
        # Delete collection and recreate it
        client.delete_collection(
            name=pipeline.chroma_manager.collection_name
        )
        collection = pipeline.chroma_manager.get_or_create_collection()
        
        count_after = collection.count()
        print(f"✅ Database reset complete.")
        print(f"   Documents: {count_before} → {count_after}")
        return True
    
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = reset_database()
    sys.exit(0 if success else 1)
