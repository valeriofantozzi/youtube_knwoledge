# AI Search Implementation Plan

## Technical & Engineering Description

### Overview

We will implement a new "AI Search" tab in the Streamlit application. This feature transforms the application from a simple semantic search tool into a conversational AI assistant (RAG Chatbot). It uses **LangGraph** to orchestrate a pipeline that optimizes user queries, retrieves relevant subtitles from the existing vector store, and generates natural language answers.

### Architecture

The solution uses a **Stateful Graph** architecture provided by LangGraph.

- **State:** A typed dictionary containing the chat history, current query, retrieved documents, and generated answer.
- **Graph Flow:** `Start` -> `Query Optimization` -> `Retrieval` -> `Answer Generation` -> `End`.

### Components & Modules

1.  **`src/ai_search/`**: New module containing the RAG logic.
    - `state.py`: Defines the `AgentState` (messages, context, etc.).
    - `graph.py`: Defines the LangGraph workflow (nodes and edges).
    - `chains.py`: Defines the LangChain runnables for Query Expansion and Answer Generation.
2.  **`src/ui/pages/ai_search_page.py`**: New UI component using Streamlit's chat elements (`st.chat_message`, `st.chat_input`) to interact with the graph.
3.  **Integration**: The existing `QueryEngine` in `src/retrieval/query_engine.py` will be used as a tool/function within the `Retrieval` node.

### Technology Stack

- **LangChain**: For prompt templates and LLM interaction.
- **LangGraph**: For building the cyclic/stateful control flow.
- **Streamlit**: For the chat interface.
- **LLM**: Configurable (defaulting to OpenAI for the plan, but compatible with others).

### Data Flow

1.  User inputs text in Streamlit.
2.  **Query Optimization Node**: LLM rewrites the query to be better suited for semantic search (e.g., removing conversational fluff, adding keywords).
3.  **Retrieval Node**: The optimized query is passed to `QueryEngine.query()`, which queries ChromaDB and returns `SearchResult` objects.
4.  **Generation Node**: The LLM receives the original question + retrieved context and generates a final answer.
5.  Response is streamed back to the Streamlit UI.

## Implementation Plan

1. [ ] Phase 1: Project Setup & Dependencies
       _Description: Prepare the environment for LangChain and LangGraph._

   1.1. [ ] Task: Update `requirements.txt`
   _Description: Add `langchain`, `langgraph`, `langchain-openai`, `langchain-community`._

   1.2. [ ] Task: Configure Environment Variables
   _Description: Ensure `.env` can handle `OPENAI_API_KEY` or equivalent model configuration._

2. [ ] Phase 2: Backend Implementation (LangGraph Pipeline)
       _Description: Implement the core RAG logic using LangGraph._

   2.1. [ ] Task: Create `src/ai_search` module structure
   _Description: Create directory and `__init__.py`._

   2.2. [ ] Task: Define Graph State (`src/ai_search/state.py`)
   _Description: Define `AgentState` TypedDict with keys for `messages`, `question`, `documents`, `generation`._

   2.3. [ ] Task: Implement LLM Chains (`src/ai_search/chains.py`)
   _Description: Create PromptTemplates and LLM chains for:_

   - **Query Rewriter**: "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history."
   - **RAG Generator**: "Answer the question based only on the following context."

   2.4. [ ] Task: Implement Graph Nodes (`src/ai_search/graph.py`)
   _Description: Implement functions for:_

   - `rewrite_query(state)`: Calls the rewriter chain.
   - `retrieve(state)`: Instantiates `QueryEngine` and calls `.query()`.
   - `generate(state)`: Calls the generator chain.

   2.5. [ ] Task: Compile the Graph
   _Description: Define the workflow, add nodes, add edges (`rewrite` -> `retrieve` -> `generate`), and compile the app._

3. [ ] Phase 3: UI Implementation
       _Description: Create the Chat Interface in Streamlit._

   3.1. [ ] Task: Create `src/ui/pages/ai_search_page.py`
   _Description: Implement `render_ai_search_page()`._

   - Initialize chat history in `st.session_state`.
   - Render chat messages using `st.chat_message`.
   - Handle user input with `st.chat_input`.
   - Invoke the LangGraph app and display the response.

   3.2. [ ] Task: Update `streamlit_app.py`
   _Description: Add "AI Search" to the sidebar navigation and route to the new page function._

4. [ ] Phase 4: Testing & Refinement
       _Description: Verify the pipeline works with real data._

   4.1. [ ] Task: Manual Testing
   _Description: Run the app, ask a question, verify the query is rewritten, documents are retrieved, and an answer is generated._

   4.2. [ ] Task: Prompt Tuning
   _Description: Adjust system prompts in `chains.py` for better performance on subtitle data._
