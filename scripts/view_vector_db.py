#!/usr/bin/env python3
"""
Simple script to view the vector database contents.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.chroma_manager import ChromaDBManager
from collections import Counter


def search_documents(collection, query: str, top_k: int = 5):
    """Search documents using semantic search."""
    from src.embeddings.embedder import Embedder
    
    print(f"\n🔍 Searching for: '{query}'\n")
    
    try:
        # Generate query embedding
        embedder = Embedder()
        query_embedding = embedder.encode([query], is_query=True)[0]
        
        # Search in collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results.get('ids') or not results['ids'][0]:
            print("⚠️  No results found")
            return
        
        # Display results
        print("=" * 70)
        print(f"📋 SEARCH RESULTS (Top {len(results['ids'][0])})")
        print("=" * 70)
        
        for i, (doc_id, doc_text, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = 1 - distance  # Convert distance to similarity
            
            print(f"\n--- Result {i} (Similarity: {similarity:.3f}) ---")
            if metadata:
                if 'video_id' in metadata:
                    print(f"Video ID: {metadata['video_id']}")
                if 'title' in metadata:
                    print(f"Title: {metadata['title']}")
                if 'date' in metadata:
                    print(f"Date: {metadata['date']}")
            print(f"Text: {doc_text[:300]}...")
    
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()


def main():
    """View vector database."""
    parser = argparse.ArgumentParser(description="View and search vector database")
    parser.add_argument("--search", type=str, help="Search query (semantic search)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample documents (default: 5)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📊 VECTOR DATABASE VIEWER")
    print("=" * 70)
    print()
    
    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    manager = ChromaDBManager()
    collection = manager.get_or_create_collection()
    
    # Get statistics
    total_docs = collection.count()
    print(f"✅ Total documents: {total_docs}")
    print()
    
    if total_docs == 0:
        print("⚠️  Database is empty!")
        return
    
    # Get sample
    sample_size = min(1000, total_docs)
    print(f"Analyzing sample of {sample_size} documents...")
    sample = collection.get(limit=sample_size)
    
    # Analyze metadata
    video_ids = []
    dates = []
    titles = []
    
    if sample.get('metadatas'):
        for meta in sample['metadatas']:
            if meta:
                if 'video_id' in meta:
                    video_ids.append(meta['video_id'])
                if 'date' in meta:
                    dates.append(meta['date'])
                if 'title' in meta:
                    titles.append(meta['title'])
    
    unique_videos = len(set(video_ids)) if video_ids else 0
    avg_chunks = total_docs / unique_videos if unique_videos > 0 else 0
    
    print()
    print("=" * 70)
    print("📊 STATISTICS")
    print("=" * 70)
    print(f"Total Documents:     {total_docs}")
    print(f"Unique Videos:       {unique_videos}")
    print(f"Avg Chunks/Video:    {avg_chunks:.1f}")
    
    if sample.get('embeddings') and sample['embeddings']:
        emb_dim = len(sample['embeddings'][0])
        print(f"Embedding Dimension: {emb_dim}")
    
    print()
    
    # Show date distribution
    if dates:
        date_counter = Counter(dates)
        print("=" * 70)
        print("📅 DATE DISTRIBUTION (Top 10)")
        print("=" * 70)
        for date, count in date_counter.most_common(10):
            print(f"  {date}: {count} chunks")
        print()
    
    # Show sample documents
    print("=" * 70)
    print(f"📄 SAMPLE DOCUMENTS (First {args.samples})")
    print("=" * 70)
    
    for i, (doc_id, doc_text, metadata) in enumerate(zip(
        sample['ids'][:args.samples],
        sample.get('documents', [''] * len(sample['ids']))[:args.samples],
        sample.get('metadatas', [{}] * len(sample['ids']))[:args.samples]
    ), 1):
        print(f"\n--- Document {i} ---")
        if metadata:
            if 'video_id' in metadata:
                print(f"Video ID: {metadata['video_id']}")
            if 'title' in metadata:
                print(f"Title: {metadata['title']}")
            if 'date' in metadata:
                print(f"Date: {metadata['date']}")
        print(f"Text: {doc_text[:200]}...")
    
    # Perform search if requested
    if args.search:
        search_documents(collection, args.search, args.top_k)
    else:
        print()
        print("=" * 70)
        print("✅ Database visualization complete!")
        print("=" * 70)
        print()
        print("💡 Tip: Use --search option for semantic search")
        print("   Example: python scripts/view_vector_db.py --search 'orchid care'")


if __name__ == "__main__":
    main()

