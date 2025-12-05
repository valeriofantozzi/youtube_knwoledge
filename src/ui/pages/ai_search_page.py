import html
import streamlit as st
from typing import cast
from langchain_core.messages import HumanMessage, AIMessage
from src.ai_search.graph import build_graph
from src.ai_search.state import AgentState
from src.ui.theme import ICONS
from src.ui.components.thinking_display import render_thinking_status_simple

def set_suggested_query(query):
    """Callback to set the suggested query."""
    st.session_state.suggested_query = query

@st.dialog("📄 Document Details", width="large")
def view_document(filename, score, text, metadata):
    st.markdown(f"### {filename}")
    st.caption(f"Relevance Score: {score}")
    # st.divider()
    
    # Collapsible content section
    with st.expander("📄 Content", expanded=True):
        st.markdown(text)
    # Collapsible metadata section (default collapsed)
    if metadata:
        with st.expander("📋 Metadata", expanded=False):
            # Convert metadata to dict if it's not already
            if isinstance(metadata, dict):
                metadata_dict = metadata
            else:
                metadata_dict = vars(metadata) if hasattr(metadata, '__dict__') else {"type": str(type(metadata))}
            st.json(metadata_dict)

def render_sources_carousel(documents):
    """
    Render retrieved documents in a horizontal scrollable carousel using styled buttons.
    """
    if not documents:
        return
    
    # st.markdown("### 📚 Sources")
    
    # Custom CSS for horizontal scrolling and styled buttons
    st.markdown(
        """
        <style>
        /* Force horizontal scrolling for the columns container */
        div[data-testid="stHorizontalBlock"] {
            overflow-x: auto;
            flex-wrap: nowrap !important;
            padding-bottom: 10px; /* Space for scrollbar */
        }
        
        /* Force minimum width for each column to ensure they don't shrink */
        div[data-testid="stColumn"] {
            min-width: 220px !important;
            flex: 0 0 auto !important; /* Don't grow or shrink */
        }

        /* Styled Buttons */
        div[data-testid="stColumn"] button {
            height: auto !important;
            min-height: 70px !important;
            padding: 10px !important;
            text-align: left !important;
            display: block !important;
            width: 100% !important;
            white-space: pre-wrap !important;
            line-height: 1.4 !important;
            border: 1px solid #e0e0e0 !important;
            background-color: #f8f9fa !important;
        }

        div[data-testid="stColumn"] button:hover {
            border-color: #FF4B4B !important;
            background-color: #fff !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        div[data-testid="stColumn"] button p {
            font-size: 0.9rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a single row with one column per document
    cols = st.columns(len(documents))

    for col, doc in zip(cols, documents):
        with col:
            # Normalize doc attributes
            if isinstance(doc, dict):
                text = doc.get("text", "")
                score = doc.get("similarity_score", 0)
                metadata = doc.get("metadata", {})
                filename = metadata.get("filename", "Unknown")
            else:
                text = doc.text
                score = doc.similarity_score
                metadata = doc.metadata
                filename = getattr(metadata, "filename", "Unknown")

            score_display = f"{score:.2f}" if score is not None else "N/A"
            safe_score = score if score is not None else 0.0
            
            # Determine icon
            if safe_score >= 0.7:
                icon = "🟢"
            elif safe_score >= 0.4:
                icon = "🟡" 
            else:
                icon = "🔴"

            truncated_filename = filename if len(filename) <= 20 else filename[:17] + "..."
            
            # Button Label: Icon + Filename \n Score
            label = f"📄 {truncated_filename}\n{icon} Score: {score_display}"
            
            if st.button(label, key=f"view_{id(doc)}", use_container_width=True):
                view_document(filename, score_display, text, metadata)

def render_ai_search_page():
    """
    Render the AI Search page with chat interface.
    """
    st.title(f"{ICONS.get('search', '🤖')} AI Search Assistant")
    st.markdown("Ask questions about your documents and get AI-generated answers based on the content.")
    
    # Add auto-scroll JavaScript
    st.markdown("""
    <script>
    window.addEventListener('load', function() {
        setTimeout(() => {
            window.scrollTo(0, document.body.scrollHeight);
        }, 100);
    });
    
    // Also scroll when mutations happen (new content added)
    const observer = new MutationObserver(() => {
        window.scrollTo(0, document.body.scrollHeight);
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: false
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize graph
    if "ai_graph" not in st.session_state:
        with st.spinner("Initializing AI Agent..."):
            try:
                st.session_state.ai_graph = build_graph()
            except Exception as e:
                st.error(f"Failed to initialize AI Agent: {e}")
                return
    
    # Handle suggested query click
    if "suggested_query" in st.session_state:
        suggested = st.session_state.pop("suggested_query")
        st.session_state.pending_query = suggested

    # Display chat messages (excluding the one being processed in this render)
    messages_to_display = st.session_state.messages
    for idx, message in enumerate(messages_to_display):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)
            
            # Render sources if available in additional_kwargs
            if isinstance(message, AIMessage):
                kwargs = message.additional_kwargs
                docs = kwargs.get("documents", [])
                needs_clarification = kwargs.get("needs_clarification", False)
                query_analysis = kwargs.get("query_analysis", {})
                
                # Only show documents if not a clarification response
                if docs and not needs_clarification:
                    render_sources_carousel(docs)

    # Capture user input from chat box (always render it)
    user_input = st.chat_input("Ask a question...")
    
    # Check for pending query from suggestions
    pending_query = st.session_state.pop("pending_query", None)
    
    # Determine which prompt to use (priority to pending_query)
    prompt = pending_query if pending_query else user_input
    
    if prompt:
        # Add user message to history
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            thinking_placeholder = st.empty()
            
            # Show initial thinking state
            with thinking_placeholder.container():
                st.markdown("⏳ Thinking...")
            
            # Initialize thinking updates for this session
            thinking_updates = []
            
            try:
                # Prepare state
                initial_state = cast(AgentState, {
                    "messages": st.session_state.messages,
                    "question": prompt,
                    "documents": [],
                    "generation": "",
                    "query_analysis": None,
                    "needs_clarification": False,
                    "clarification_response": None,
                    "thinking_updates": []
                })
                
                # Collect all updates as we stream
                # Initialize variables to capture state across steps
                answer = ""
                documents = []
                query_analysis = {}
                needs_clarification = False
                thinking_updates = []
                response = {}
                
                # Stream the graph execution to capture thinking updates in real-time
                for chunk in st.session_state.ai_graph.stream(initial_state):
                    # Each chunk is a dict with node name as key
                    for node_name, node_state in chunk.items():
                        # Collect thinking updates from each node
                        if "thinking_updates" in node_state:
                            new_updates = node_state.get("thinking_updates", [])
                            # Only add updates we haven't seen yet
                            if len(new_updates) > len(thinking_updates):
                                thinking_updates = new_updates
                                # Update display with latest thinking status
                                with thinking_placeholder.container():
                                    render_thinking_status_simple(thinking_updates)
                        
                        # Capture other state updates as they happen
                        if "documents" in node_state and node_state["documents"]:
                            documents = node_state["documents"]
                        
                        if "query_analysis" in node_state and node_state["query_analysis"]:
                            query_analysis = node_state["query_analysis"]
                            
                        if "needs_clarification" in node_state:
                            needs_clarification = node_state["needs_clarification"]
                            
                        if "generation" in node_state:
                            answer = node_state["generation"]
                    
                    response = chunk
                
                # If no answer was generated yet (e.g. error or empty), set default
                if not answer:
                    answer = "I couldn't generate an answer."
                
                # Display answer
                message_placeholder.markdown(answer)
                
                # Clear thinking display after we have the answer
                thinking_placeholder.empty()
                
                # Force scroll to bottom
                st.markdown("""
                <script>
                    setTimeout(() => {
                        window.scrollTo(0, document.body.scrollHeight);
                    }, 100);
                </script>
                """, unsafe_allow_html=True)
                
                # Show suggested queries if clarification was needed
                if needs_clarification and query_analysis:
                    suggested = query_analysis.get("suggested_queries", [])
                    if suggested:
                        st.markdown("---")
                        st.markdown("**💡 Try asking:**")
                        cols = st.columns(min(len(suggested), 3))
                        for idx, suggestion in enumerate(suggested[:3]):
                            with cols[idx]:
                                # Use a key that matches what will be in history (len(messages) is the index of this new message)
                                msg_idx = len(st.session_state.messages)
                                st.button(
                                    f"📝 {suggestion}", 
                                    key=f"hist_suggest_{msg_idx}_{idx}",
                                    on_click=set_suggested_query,
                                    args=(suggestion,),
                                    use_container_width=True
                                )
                
                # Render sources for the new message (only if we got actual results)
                if documents and not needs_clarification:
                    render_sources_carousel(documents)
                
                # Add AI message to history with documents
                st.session_state.messages.append(AIMessage(
                    content=answer, 
                    additional_kwargs={
                        "documents": documents,
                        "needs_clarification": needs_clarification,
                        "query_analysis": query_analysis,
                        "thinking_updates": thinking_updates
                    }
                ))
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
                # Still display thinking updates even if there was an error
                if thinking_updates:
                    with thinking_placeholder.container():
                        render_thinking_status_simple(thinking_updates)
