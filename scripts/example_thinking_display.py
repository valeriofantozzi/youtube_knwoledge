"""
Example: Agent Thinking Display System

This example demonstrates how the new thinking display system works.
Run this to see the thinking updates in action.

Usage:
    python scripts/example_thinking_display.py
"""

from src.ai_search.thinking import (
    ThinkingEmitter,
    ThinkingStatus,
    ThinkingSession,
    ThinkingUpdate
)
import json


def example_1_basic_updates():
    """Example 1: Creating and displaying basic thinking updates."""
    print("=" * 60)
    print("Example 1: Basic Thinking Updates")
    print("=" * 60)
    
    emitter = ThinkingEmitter("Query Analyzer")
    
    # Emit several updates
    updates = [
        emitter.emit_analyzing("Analyzing query", progress=0.2),
        emitter.emit_processing("Extracting entities", progress=0.5),
        emitter.emit_complete("Analysis complete"),
    ]
    
    # Display as JSON
    for update in updates:
        print(f"\n{update.agent_name}: {update.phase_title}")
        print(f"  Status: {update.status.value}")
        print(f"  Progress: {update.progress}")
        print(f"  Details: {update.details}")


def example_2_multiple_agents():
    """Example 2: Thinking session with multiple agents."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Agents Processing")
    print("=" * 60)
    
    session = ThinkingSession(query="How do orchids grow?")
    
    # Query Analyzer
    qa_emitter = ThinkingEmitter("Query Analyzer")
    session.add_update(qa_emitter.emit_analyzing("Analyzing clarity", progress=0.3))
    session.add_update(qa_emitter.emit_processing("Checking intent", progress=0.7))
    session.add_update(qa_emitter.emit_complete("Query is clear"))
    
    # Rewriter
    qr_emitter = ThinkingEmitter("Query Rewriter")
    session.add_update(qr_emitter.emit_processing("Normalizing query", progress=0.5))
    session.add_update(qr_emitter.emit_complete("Query rewritten"))
    
    # Retriever
    ret_emitter = ThinkingEmitter("Document Retriever")
    session.add_update(ret_emitter.emit_retrieving("Searching knowledge base", progress=0.4))
    session.add_update(ret_emitter.emit_processing("Ranking documents", progress=0.8))
    session.add_update(ret_emitter.emit_complete(
        "Retrieved documents",
        metadata={"documents": 5, "total_score": 4.2}
    ))
    
    # Generator
    gen_emitter = ThinkingEmitter("Answer Generator")
    session.add_update(gen_emitter.emit_reasoning("Analyzing sources", progress=0.3))
    session.add_update(gen_emitter.emit_generating("Creating answer", progress=0.8))
    session.add_update(gen_emitter.emit_complete(
        "Answer generated",
        metadata={"sources_used": 5, "tokens": 256}
    ))
    
    # Display summary
    print(f"\nQuery: {session.query}")
    print(f"Total updates: {len(session.updates)}")
    print(f"Agents involved: {len(session.get_agent_updates.__code__.co_freevars)}")
    
    # Group by agent
    agents = {}
    for update in session.updates:
        if update.agent_name not in agents:
            agents[update.agent_name] = []
        agents[update.agent_name].append(update)
    
    for agent_name, agent_updates in agents.items():
        print(f"\n{agent_name}:")
        for update in agent_updates:
            print(f"  → {update.phase_title} ({update.status.value})")


def example_3_metadata_and_errors():
    """Example 3: Using metadata and error handling."""
    print("\n" + "=" * 60)
    print("Example 3: Metadata and Error Handling")
    print("=" * 60)
    
    emitter = ThinkingEmitter("Search Agent")
    
    # Normal processing
    updates = [
        emitter.emit_retrieving(
            "Searching vector database",
            details="Using cosine similarity",
            progress=0.5,
            metadata={
                "query_embedding_dim": 768,
                "database_size": 10000,
                "similarity_threshold": 0.7
            }
        ),
        emitter.emit_complete(
            "Search complete",
            metadata={
                "results_found": 42,
                "top_score": 0.92,
                "search_time_ms": 125
            }
        ),
    ]
    
    # Error case
    error_update = emitter.emit_error(
        "Search failed",
        details="Connection timeout after 30s",
        metadata={
            "error_type": "TimeoutError",
            "retry_count": 3
        }
    )
    updates.append(error_update)
    
    # Display as JSON
    print("\nUpdates as JSON:")
    print(json.dumps([u.to_dict() for u in updates], indent=2))


def example_4_streamlit_integration():
    """Example 4: How this integrates with Streamlit."""
    print("\n" + "=" * 60)
    print("Example 4: Streamlit Integration")
    print("=" * 60)
    
    print("""
In your Streamlit app:

1. Initialize thinking updates in state:
   initial_state = {
       "question": "How do orchids bloom?",
       "thinking_updates": [],  # Collect updates here
       ...
   }

2. Run graph (agents emit updates):
   response = graph.invoke(initial_state)
   thinking_updates = response.get("thinking_updates", [])

3. Display in UI:
   from src.ui.components.thinking_display import render_thinking_inline
   render_thinking_inline(thinking_updates)

4. Output:
   🔍 Analyzing query clarity and intent
      Query Analyzer: Evaluating question: 'How do orchids bloom?'
   
   📚 Searching knowledge base
      Document Retriever: Finding relevant documents...
   
   ✅ Document retrieval complete
      Document Retriever: Found 5 relevant documents
   
   ... and so on
    """)


if __name__ == "__main__":
    example_1_basic_updates()
    example_2_multiple_agents()
    example_3_metadata_and_errors()
    example_4_streamlit_integration()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
