"""
Quick Reference: Agent Thinking Display System

Copy-paste examples for common use cases.
"""

# ============================================================================

# EXAMPLE 1: Using ThinkingEmitter in your agent node

# ============================================================================

from src.ai_search.thinking import ThinkingEmitter
from src.ai_search.state import AgentState

def my_custom_agent(state: AgentState):
"""Example agent node that emits thinking updates."""

    # 1. Create emitter with your agent's name
    emitter = ThinkingEmitter("My Custom Agent")

    # 2. Get existing updates from state (or initialize)
    thinking_updates = state.get("thinking_updates", [])

    # 3. Emit analyzing phase
    thinking_updates.append(
        emitter.emit_analyzing(
            "Analyzing input data",
            details="Processing 50 documents...",
            progress=0.2
        )
    )

    # Do some work...

    # 4. Emit processing phase
    thinking_updates.append(
        emitter.emit_processing(
            "Applying transformations",
            details="Using pipeline v2.1...",
            progress=0.5,
            metadata={"pipeline_version": "2.1", "gpu_memory_mb": 4096}
        )
    )

    # Do more work...

    # 5. Emit complete with metadata
    thinking_updates.append(
        emitter.emit_complete(
            "Processing complete",
            details="Generated 50 embeddings successfully",
            metadata={
                "embeddings_count": 50,
                "model": "all-MiniLM-L6-v2",
                "processing_time_seconds": 2.34
            }
        )
    )

    # 6. Or emit error if something fails
    try:
        # ... do something risky ...
        pass
    except Exception as e:
        thinking_updates.append(
            emitter.emit_error(
                "Processing failed",
                details=str(e),
                metadata={"error_type": type(e).__name__}
            )
        )

    # 7. Return state with updated thinking_updates
    return {
        "result": "my result",
        "thinking_updates": thinking_updates
    }

# ============================================================================

# EXAMPLE 2: Display thinking updates in Streamlit

# ============================================================================

import streamlit as st
from src.ui.components.thinking_display import (
render_thinking_inline,
render_thinking_expandable,
render_thinking_session
)

# Option A: Inline display (best for chat)

def display_thinking_inline(thinking_updates):
"""Show thinking updates inline between messages."""
st.markdown("---")
st.markdown("### 🧠 Agent Thinking Process")
render_thinking_inline(thinking_updates)
st.markdown("---")

# Option B: Expandable display (saves space)

def display_thinking_expandable(thinking_updates):
"""Show thinking in collapsible section."""
render_thinking_expandable(thinking_updates)

# Option C: Tab-based display (for multiple agents)

def display_thinking_tabs(thinking_updates):
"""Show thinking grouped by agent in tabs."""
render_thinking_session(thinking_updates)

# ============================================================================

# EXAMPLE 3: Complete workflow integration

# ============================================================================

def complete_workflow_example():
"""
Full example showing how thinking updates flow through the system.
"""

    from langchain_core.messages import HumanMessage
    from src.ai_search.graph import build_graph

    # 1. Initialize thinking updates in state
    initial_state = {
        "messages": [HumanMessage(content="How do orchids grow?")],
        "question": "How do orchids grow?",
        "documents": [],
        "generation": "",
        "query_analysis": None,
        "needs_clarification": False,
        "clarification_response": None,
        "thinking_updates": []  # ← Initialize empty
    }

    # 2. Run the graph (agents emit updates)
    graph = build_graph()
    response = graph.invoke(initial_state)

    # 3. Extract results
    answer = response.get("generation", "")
    thinking_updates = response.get("thinking_updates", [])  # ← Collect updates

    # 4. Display in UI
    st.write(answer)

    if thinking_updates:
        display_thinking_inline(thinking_updates)

    # 5. Store in history for replay
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "thinking_updates": thinking_updates
    })

# ============================================================================

# EXAMPLE 4: Custom status types in your agent

# ============================================================================

from src.ai_search.thinking import ThinkingEmitter, ThinkingStatus

def agent_with_custom_flow(state: AgentState):
"""Agent using different status types."""

    emitter = ThinkingEmitter("Custom Agent")
    updates = state.get("thinking_updates", [])

    # Different status flows based on agent type:

    # For retrieval agents
    updates.append(emitter.emit_retrieving("Searching database", progress=0.5))
    updates.append(emitter.emit_complete("Retrieved 10 documents"))

    # For reasoning agents
    updates.append(emitter.emit_reasoning("Analyzing results", progress=0.4))
    updates.append(emitter.emit_generating("Forming answer", progress=0.7))
    updates.append(emitter.emit_complete("Answer ready"))

    # For validation agents
    updates.append(emitter.emit_processing("Validating output", progress=0.5))
    updates.append(emitter.emit_complete("Validation passed"))

    return {"thinking_updates": updates}

# ============================================================================

# EXAMPLE 5: Metadata for debugging and analytics

# ============================================================================

def agent_with_rich_metadata(state: AgentState):
"""Agent that includes detailed metadata for debugging."""

    import time
    emitter = ThinkingEmitter("Analytics Agent")
    updates = state.get("thinking_updates", [])

    start_time = time.time()

    updates.append(
        emitter.emit_analyzing(
            "Analyzing query patterns",
            metadata={
                "sample_size": 1000,
                "features": ["length", "sentiment", "topic"],
                "model_name": "query-analyzer-v3"
            }
        )
    )

    # ... process ...

    elapsed = time.time() - start_time

    updates.append(
        emitter.emit_complete(
            "Analysis complete",
            metadata={
                "patterns_found": 47,
                "confidence_score": 0.92,
                "processing_time_seconds": round(elapsed, 2),
                "memory_used_mb": 256,
                "cache_hits": 12,
                "cache_misses": 3
            }
        )
    )

    return {"thinking_updates": updates}

# ============================================================================

# EXAMPLE 6: Error handling and recovery

# ============================================================================

def agent_with_error_recovery(state: AgentState):
"""Agent that handles errors gracefully."""

    emitter = ThinkingEmitter("Resilient Agent")
    updates = state.get("thinking_updates", [])

    max_retries = 3

    for attempt in range(max_retries):
        try:
            updates.append(
                emitter.emit_processing(
                    f"Attempt {attempt + 1}/{max_retries}",
                    details="Trying to connect to service...",
                    progress=0.3 * (attempt + 1)
                )
            )

            # Try something risky
            result = some_risky_operation()

            updates.append(emitter.emit_complete("Success"))
            break

        except Exception as e:
            if attempt < max_retries - 1:
                updates.append(
                    emitter.emit_error(
                        f"Attempt {attempt + 1} failed, retrying...",
                        details=str(e),
                        metadata={"attempt": attempt + 1, "error_type": type(e).__name__}
                    )
                )
            else:
                updates.append(
                    emitter.emit_error(
                        "All retry attempts failed",
                        details=str(e),
                        metadata={"total_attempts": max_retries}
                    )
                )
                # Fall back to default behavior

    return {"thinking_updates": updates}

# ============================================================================

# EXAMPLE 7: Conditional display based on user settings

# ============================================================================

def conditional_thinking_display(thinking_updates, show_details=False):
"""Show thinking with optional details expansion."""

    if not thinking_updates:
        st.info("No thinking updates captured")
        return

    # Summary view
    st.markdown("### 🧠 Agent Thinking")
    for update in thinking_updates:
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            st.markdown(f"{update.status.value}")
        with col2:
            st.markdown(f"**{update.phase_title}**")
            if update.details:
                st.caption(update.details)

    # Detailed view
    if show_details:
        st.markdown("---")
        st.markdown("### 📊 Detailed Metadata")
        for update in thinking_updates:
            if update.metadata:
                with st.expander(f"{update.agent_name}: {update.phase_title}"):
                    st.json(update.metadata)

# ============================================================================

# TIPS AND BEST PRACTICES

# ============================================================================

"""
BEST PRACTICES:

1. Always initialize thinking_updates:
   ✓ thinking_updates = state.get("thinking_updates", [])
   ✗ thinking_updates = state["thinking_updates"] # May not exist

2. Use appropriate status types:
   ✓ ANALYZING for initial input assessment
   ✓ PROCESSING for transformations
   ✓ RETRIEVING for knowledge base searches
   ✓ GENERATING for creating outputs
   ✓ REASONING for decision making
   ✓ COMPLETE for success
   ✓ ERROR for failures

3. Include meaningful phase titles:
   ✓ "Analyzing query clarity and intent"
   ✗ "Processing..."

4. Use progress for long operations:
   ✓ 0.0 = just started
   ✓ 0.5 = halfway done
   ✓ 1.0 = complete (use emit_complete() instead)

5. Add metadata for debugging:
   ✓ metadata={"items_processed": 100, "model": "v2"}
   ✓ metadata={"error_type": type(e).**name**, "retry_count": 3}

6. Always return thinking_updates in your node:
   ✓ return {"result": ..., "thinking_updates": thinking_updates}
   ✗ return {"result": ...} # Loses thinking info

7. Display thinking between messages:
   ✓ Show after user message, before AI response
   ✗ Show after the response (user already sees answer)

8. Keep update count reasonable:
   ✓ 5-10 updates per agent is typical
   ✗ 100+ updates per agent (too much noise)
   """
