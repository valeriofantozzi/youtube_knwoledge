# Testing & Validation Guide

## Quick Start Testing

### 1. Run Example Script

```bash
cd /Users/valeriofantozzi/Developer/youtube_kwoledge
python scripts/example_thinking_display.py
```

Expected output:

- Example 1: Shows individual thinking updates
- Example 2: Shows multi-agent session
- Example 3: Shows metadata and error handling
- Example 4: Shows Streamlit integration pattern

### 2. Test in Streamlit App

```bash
cd /Users/valeriofantozzi/Developer/youtube_kwoledge
streamlit run streamlit_app.py
```

Then:

1. Navigate to "🤖 AI Search" tab
2. Enter a query like: "How do orchids grow?"
3. Observe the agent thinking process displayed dynamically

Expected behavior:

- Shows "Agent Thinking Process" section
- Each agent appears with its status icon and phase title
- Progress bars for ongoing phases
- Detailed metadata visible in expandable sections
- Final answer and sources displayed below

### 3. Test Error Handling

Query that might trigger analysis issues:

```
test
```

Expected: Should still show thinking updates even if clarification needed

## Code Validation Checklist

- [ ] All imports resolve correctly
- [ ] No type errors in thinking.py
- [ ] No type errors in thinking_display.py
- [ ] AgentState includes thinking_updates field
- [ ] All graph nodes return thinking_updates
- [ ] ai_search_page initializes thinking_updates
- [ ] Thinking display component renders without errors

## Manual Testing Checklist

### Basic Functionality

- [ ] Query analysis shows update flow
- [ ] Status icons display correctly
- [ ] Agent names are visible
- [ ] Progress bars animate
- [ ] Details section expands/collapses

### Multiple Agents

- [ ] Multiple agents shown in sequence
- [ ] Each agent has its own updates
- [ ] Status progression is logical (ANALYZING → PROCESSING → COMPLETE)

### Error Cases

- [ ] Empty query shows error
- [ ] Network timeout shows error update
- [ ] Partial processing shows partial updates
- [ ] Error details are informative

### Display Modes

- [ ] Inline mode works (default)
- [ ] Expandable mode works
- [ ] Tab mode works (multi-agent)
- [ ] Stream mode works

### Metadata

- [ ] Metadata displays in JSON expander
- [ ] Metadata is JSON-serializable
- [ ] Progress values range 0.0-1.0
- [ ] Timestamps are ISO format

## Performance Testing

### Time Measurements

```python
import time

# Measure overhead of thinking updates
start = time.time()
# Run a query
end = time.time()

# Expected: < 1 second additional overhead
```

### Memory Testing

```python
import tracemalloc

tracemalloc.start()
# Run queries with thinking updates
current, peak = tracemalloc.get_traced_memory()
# Expected: < 1 MB additional memory per session
```

## Regression Testing

After updates, ensure:

1. **Graph still works**: Query processing completes successfully
2. **UI still renders**: No Streamlit errors
3. **State management**: State flows correctly through nodes
4. **Error recovery**: Errors are handled gracefully
5. **Historical messages**: Old messages still display correctly

## Integration Test Cases

### Test Case 1: Simple Query

```
Input: "How do orchids bloom?"
Expected:
- Query Analyzer: ANALYZING → COMPLETE
- Query Rewriter: PROCESSING → COMPLETE
- Document Retriever: RETRIEVING → PROCESSING → COMPLETE
- Answer Generator: REASONING → GENERATING → COMPLETE
```

### Test Case 2: Vague Query

```
Input: "tell me about plants"
Expected:
- Query Analyzer: ANALYZING → COMPLETE (needs_clarification=True)
- Clarification Agent: GENERATING → COMPLETE
- No retrieval/generation
```

### Test Case 3: Multi-turn Conversation

```
Input 1: "How do orchids grow?"
[Agent updates shown]

Input 2: "What about watering?"
[Agent updates shown for follow-up]

Expected: Each query shows fresh thinking updates
```

### Test Case 4: Error in Processing

```
Input: [Invalid query that causes error]
Expected:
- Partial thinking updates shown
- Error update appears
- User sees error message
- Thinking shown up to failure point
```

## Browser Developer Tools Check

1. Open DevTools (F12)
2. Go to Console
3. Check for JavaScript errors
4. Go to Network tab
5. Verify reasonable response times (< 30s typical)

## Accessibility Testing

- [ ] Screen reader friendly (semantic HTML)
- [ ] Color not only indicator of status (use icons too)
- [ ] Progress text visible (not just bar)
- [ ] Metadata visible in readable format
- [ ] Good contrast between text and background

## Cross-Browser Testing

- [ ] Chrome/Edge (Chromium-based)
- [ ] Firefox
- [ ] Safari
- [ ] Mobile browsers (iPhone, Android)

## Load Testing

```python
# Test with rapid queries
for i in range(10):
    query = f"Test query {i}"
    # Submit query
    # Verify thinking updates
```

Expected: System handles multiple queries gracefully

## Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run queries and check console for:
# - Graph execution flow
# - Node outputs
# - Thinking update emission
```

## JSON Validation

All thinking updates must be JSON-serializable:

```python
import json
from src.ai_search.thinking import ThinkingUpdate

update = ThinkingUpdate(...)
json_str = update.to_json()
parsed = json.loads(json_str)
# If no exception, JSON is valid
```

## State Consistency Check

```python
from src.ai_search.state import AgentState

# Verify AgentState includes thinking_updates
assert "thinking_updates" in AgentState.__annotations__

# Verify it's a List
from typing import get_args
field_type = AgentState.__annotations__["thinking_updates"]
# Should be List[Any]
```

## Storage Testing

If persisting thinking updates:

```python
# Test JSON serialization
thinking_updates = response.get("thinking_updates", [])
serialized = json.dumps([u.to_dict() for u in thinking_updates])

# Test deserialization
deserialized = json.loads(serialized)

# Verify roundtrip
assert len(deserialized) == len(thinking_updates)
```

## Performance Baseline

Record baseline performance metrics:

| Operation                 | Time (ms) | Memory (MB) |
| ------------------------- | --------- | ----------- |
| Query analysis            | 100-200   | 1-2         |
| Document retrieval        | 500-1000  | 5-10        |
| Answer generation         | 1000-3000 | 10-20       |
| Thinking updates overhead | 1-10      | 0.1-0.5     |
| **Total query**           | 1500-4200 | 15-30       |

After changes, compare against baseline.

## Troubleshooting Guide

### Issue: Thinking updates not showing

**Check:**

- [ ] thinking_updates initialized in state
- [ ] Graph node returns thinking_updates
- [ ] Display component imported correctly
- [ ] No errors in Streamlit console

**Fix:**

```python
# Verify state initialization
print("thinking_updates" in initial_state)  # Should be True

# Verify graph returns it
response = graph.invoke(initial_state)
print(response.get("thinking_updates"))  # Should be list, not None
```

### Issue: Progress bars not animating

**Check:**

- [ ] Progress value is float 0.0-1.0
- [ ] Multiple updates with increasing progress
- [ ] render_thinking_inline imported

**Fix:**

```python
# Verify progress
update.progress >= 0.0 and update.progress <= 1.0
```

### Issue: Metadata not visible

**Check:**

- [ ] Metadata dict is not empty
- [ ] Values are JSON-serializable
- [ ] using `render_thinking_session()` for expanders

**Fix:**

```python
# Test JSON serialization
import json
json.dumps(update.metadata)  # Should not raise exception
```

### Issue: Too many updates

**Check:**

- [ ] Agents emitting excessive updates
- [ ] Each agent should have 3-5 updates typical

**Fix:**

```python
# Consolidate updates
# Instead of: analyzing each step
# Do: one update per logical phase
```

## Success Criteria

✅ All tests pass when:

1. Thinking updates emit for all agents
2. Display renders without errors
3. Progress tracks from start to finish
4. Metadata is accessible and useful
5. Errors show even on failure
6. Performance overhead < 10%
7. No memory leaks after 100 queries
8. Historical messages still work
9. Works on mobile/small screens
10. Accessible to screen readers
