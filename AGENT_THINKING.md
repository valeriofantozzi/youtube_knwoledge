# Agent Thinking Display System

## Overview

The Agent Thinking Display system provides real-time, structured visibility into the reasoning process of AI agents as they process user queries. Instead of a static "Thinking..." spinner, users now see dynamic updates from each agent describing what it's doing at each step.

## Architecture

### Components

#### 1. **Thinking Module** (`src/ai_search/thinking.py`)

Core data structures and emitters for agent thinking updates:

- **`ThinkingStatus`**: Enum defining status types
  - `ANALYZING` - Examining input
  - `PROCESSING` - Performing transformations
  - `RETRIEVING` - Searching knowledge base
  - `GENERATING` - Creating responses
  - `REASONING` - Evaluating information
  - `COMPLETE` - Successfully finished
  - `ERROR` - Failed or errored

- **`ThinkingUpdate`**: Dataclass representing a single thinking update
  - `agent_name`: Which agent emitted the update
  - `status`: Current status type
  - `phase_title`: Human-readable description of what's happening
  - `details`: Additional context or parameters
  - `progress`: 0.0-1.0 progress indicator
  - `metadata`: Extra data (JSON-serializable dict)
  - `timestamp`: When the update was created

- **`ThinkingSession`**: Tracks complete thinking session
  - Collects all updates from all agents
  - Provides grouping by agent
  - Timestamps start and end

- **`ThinkingEmitter`**: Helper class for agents to emit updates
  ```python
  emitter = ThinkingEmitter("Query Analyzer")
  update = emitter.emit_analyzing("Analyzing clarity", details="...", progress=0.3)
  update = emitter.emit_complete("Analysis done")
  ```

#### 2. **Display Component** (`src/ui/components/thinking_display.py`)

Streamlit UI components for rendering thinking updates:

- **`render_thinking_update()`**: Renders a single update with icon, color, agent name
- **`render_thinking_session()`**: Renders all updates grouped by agent with tabs
- **`render_thinking_stream()`**: Renders updates in streaming format
- **`render_thinking_expandable()`**: Renders in collapsible section
- **`render_thinking_inline()`**: Renders inline without expander

#### 3. **Agent State** (`src/ai_search/state.py`)

Extended `AgentState` TypedDict includes:

```python
"thinking_updates": List[Any]  # Accumulates updates as agents process
```

#### 4. **Graph Nodes** (`src/ai_search/graph.py`)

All agent nodes now emit thinking updates:

- **`analyze_query()`** - Query Analyzer agent
  - Emits: `ANALYZING` → `PROCESSING` → `COMPLETE` or `ERROR`
- **`generate_clarification()`** - Clarification Agent
  - Emits: `GENERATING` → `COMPLETE`
- **`rewrite_query()`** - Query Rewriter agent
  - Emits: `PROCESSING` → `COMPLETE`
- **`retrieve()`** - Document Retriever agent
  - Emits: `RETRIEVING` → `PROCESSING` → `COMPLETE`
- **`generate()`** - Answer Generator agent
  - Emits: `REASONING` → `GENERATING` → `COMPLETE`

#### 5. **UI Integration** (`src/ui/pages/ai_search_page.py`)

AI Search page now:

- Initializes `thinking_updates` in state
- Captures updates from graph execution
- Displays thinking process between user message and assistant response
- Shows thinking even if an error occurs

## Usage Flow

### User Perspective

```
User: "How do orchids bloom?"

[Chat message appears]

Agent Thinking Process
  🔍 Analyzing query clarity and intent
     Query Analyzer: Evaluating question: 'How do orchids bloom?'

  ⚙️ Rewriting query for clarity
     Query Rewriter: Optimized question ready for retrieval

  📚 Searching knowledge base
     Document Retriever: Finding relevant documents...

  🧠 Reasoning over retrieved documents
     Answer Generator: Analyzing document relevance...

  ✍️ Generating comprehensive answer
     Answer Generator: Synthesizing information from sources...

  ✅ Answer generation complete
     Answer Generator: Synthesized response from 5 sources

---

Assistant Response...

[Sources carousel]
```

### Agent Implementation

When implementing an agent/node, use the `ThinkingEmitter`:

```python
from src.ai_search.thinking import ThinkingEmitter

def my_agent_node(state: AgentState):
    emitter = ThinkingEmitter("My Agent Name")
    thinking_updates = state.get("thinking_updates", [])

    # Emit analyzing
    update = emitter.emit_analyzing(
        "Analyzing input data",
        details="Processing 100 items...",
        progress=0.2
    )
    thinking_updates.append(update)

    # ... do work ...

    # Emit processing
    update = emitter.emit_processing(
        "Computing embeddings",
        details="Using model XYZ...",
        progress=0.6
    )
    thinking_updates.append(update)

    # ... more work ...

    # Emit complete
    update = emitter.emit_complete(
        "Processing complete",
        details="Generated 100 embeddings",
        metadata={"count": 100}
    )
    thinking_updates.append(update)

    return {
        "result": final_result,
        "thinking_updates": thinking_updates
    }
```

## Customization

### Status Icons and Colors

Edit `get_status_icon()` and `get_status_color()` in `thinking_display.py`:

```python
def get_status_icon(status: ThinkingStatus) -> str:
    icons = {
        ThinkingStatus.ANALYZING: "🔍",
        ThinkingStatus.PROCESSING: "⚙️",
        # ...
    }
    return icons.get(status, "⏳")
```

### Display Modes

Choose how to display thinking updates:

1. **Inline** (default in AI Search):

   ```python
   render_thinking_inline(thinking_updates)
   ```

2. **Expandable**:

   ```python
   render_thinking_expandable(thinking_updates)
   ```

3. **Tab-based** (for multi-agent):
   ```python
   render_thinking_session(thinking_updates)
   ```

### Adding Metadata

Include extra data in thinking updates:

```python
update = emitter.emit_generating(
    "Creating response",
    metadata={
        "model": "gpt-4",
        "temperature": 0.7,
        "tokens_used": 1234
    }
)
```

## Storage and History

Thinking updates are stored in:

- **Session state**: `st.session_state.messages[i].additional_kwargs["thinking_updates"]`
- **UI state**: `st.session_state.ai_search_thinking_updates`

This allows:

- Replaying thinking process for past messages
- Analyzing agent behavior
- Debugging failed queries

## Performance Considerations

1. **Minimal overhead**: Emitting updates is just dict/list operations
2. **Memory efficient**: Updates cleaned up with session
3. **Serializable**: All updates can be JSON-serialized for storage/logging
4. **Streaming ready**: Can be easily adapted for streaming display using callbacks

## Future Enhancements

- [ ] Real-time streaming with LangGraph callbacks
- [ ] Thinking update persistence to database
- [ ] Export thinking logs for analysis
- [ ] Custom agent metrics dashboard
- [ ] Thinking process comparison between runs
- [ ] Agent performance analytics
