#!/usr/bin/env python3
"""
Debug script to analyze duplicate detection in the database.

This script checks:
1. What documents have the same source_id hash
2. Whether the deduplication query works
3. Why duplicates might still exist
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store.chroma_manager import ChromaDBManager
from src.utils.logger import get_default_logger

logger = get_default_logger()


def analyze_duplicates():
    """Analyze duplicate patterns in the database."""
    
    print("\n" + "="*80)
    print("DUPLICATE DETECTION ANALYSIS")
    print("="*80)
    
    # Initialize database
    manager = ChromaDBManager()
    collection = manager.get_or_create_collection()
    
    # Get all documents
    print("\n🔍 Retrieving all documents from database...")
    try:
        all_docs = collection.get(include=["metadatas", "documents"])
        print(f"✓ Found {len(all_docs['ids'])} total documents")
    except Exception as e:
        print(f"✗ Error retrieving documents: {e}")
        return
    
    # Group by source_id hash (last 16 chars)
    source_id_groups = defaultdict(list)
    hash_groups = defaultdict(list)
    
    print("\n📊 Analyzing source_ids...")
    for i, (doc_id, meta) in enumerate(zip(all_docs['ids'], all_docs['metadatas'])):
        source_id = meta.get('source_id', 'UNKNOWN')
        chunk_index = meta.get('chunk_index', -1)
        filename = meta.get('filename', 'UNKNOWN')
        
        source_id_groups[source_id].append({
            'doc_id': doc_id,
            'chunk_index': chunk_index,
            'filename': filename,
            'source_id': source_id
        })
        
        # Extract hash (last 16 chars after underscore)
        parts = source_id.rsplit('_', 1)
        if len(parts) == 2:
            content_hash = parts[1]
        else:
            content_hash = 'NO_HASH'
        
        hash_groups[content_hash].append({
            'source_id': source_id,
            'chunk_index': chunk_index,
            'filename': filename,
            'doc_id': doc_id
        })
    
    # Report on hash duplicates
    print("\n📌 Documents grouped by CONTENT HASH:")
    print("-" * 80)
    
    duplicate_hashes = {h: docs for h, docs in hash_groups.items() if len(docs) > 1}
    
    if duplicate_hashes:
        print(f"⚠️  Found {len(duplicate_hashes)} content hashes used by multiple documents!\n")
        
        for content_hash, docs in sorted(duplicate_hashes.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"Hash: {content_hash}")
            print(f"  Count: {len(docs)} documents with same hash")
            for doc in docs:
                print(f"    • {doc['filename']}")
                print(f"      source_id: {doc['source_id']}")
                print(f"      chunk_index: {doc['chunk_index']}")
            print()
    else:
        print("✓ No duplicate hashes found (good sign!)")
    
    # Test the deduplication query
    print("\n🧪 Testing deduplication query logic:")
    print("-" * 80)
    
    if source_id_groups:
        # Pick first source_id with multiple chunks
        test_source_id = None
        test_chunk_index = None
        
        for source_id, docs in source_id_groups.items():
            if len(docs) > 1:
                test_source_id = source_id
                test_chunk_index = docs[0]['chunk_index']
                break
        
        if test_source_id:
            print(f"\nTesting query with:")
            print(f"  source_id: {test_source_id}")
            print(f"  chunk_index: {test_chunk_index}\n")
            
            # Try the exact query used in indexer
            try:
                result = collection.get(
                    where={
                        "$and": [
                            {"source_id": {"$eq": test_source_id}},
                            {"chunk_index": {"$eq": test_chunk_index}}
                        ]
                    },
                    include=["metadatas"]
                )
                
                print(f"✓ Query returned {len(result['ids'])} documents")
                if result['ids']:
                    for meta in result['metadatas']:
                        print(f"  • {meta.get('filename', 'UNKNOWN')}")
                else:
                    print("  (empty result)")
            
            except Exception as e:
                print(f"✗ Query failed: {e}")
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("-" * 80)
    print(f"Total documents in DB: {len(all_docs['ids'])}")
    print(f"Unique source_ids: {len(source_id_groups)}")
    print(f"Unique content hashes: {len(hash_groups)}")
    print(f"Documents with duplicate hashes: {sum(len(docs) for docs in duplicate_hashes.values())}")
    
    if duplicate_hashes:
        print(f"\n⚠️  ACTION REQUIRED:")
        print(f"   Duplicate content detected. Run:")
        print(f"   python scripts/reset_database.py")
        print(f"   knowbase load --input ./subtitles")
    else:
        print(f"\n✅ Database is clean (no duplicate hashes)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_duplicates()
