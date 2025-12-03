import html
import streamlit as st
from typing import cast
from langchain_core.messages import HumanMessage, AIMessage
from src.ai_search.graph import build_graph
from src.ai_search.state import AgentState
from src.ui.theme import ICONS

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
    Render retrieved documents using Streamlit native components.
    """
    if not documents:
        return
    
    # Add custom CSS for fixed button dimensions
    st.markdown(
        """
        <style>
        .source-btn button {
            height: 60px !important;
            min-height: 60px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    cols = st.columns(len(documents))
    
    for col, doc in zip(cols, documents):
        with col:
            # Handle both object and dict access
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
            
            # Determine icon based on score
            if score is not None and score >= 0.7:
                score_icon = "🟢"
            elif score is not None and score >= 0.4:
                score_icon = "🟡"
            else:
                score_icon = "🔴"
            
            # Combine score and filename in one button
            truncated_filename = filename if len(filename) <= 40 else filename[:30] + "..."
            label = f"{score_icon} {truncated_filename}"
            st.markdown(f'<div class="source-btn">', unsafe_allow_html=True)
            if st.button(label, key=f"btn_{id(doc)}", help=filename, use_container_width=True):
                view_document(filename, score_display, text, metadata)
            st.markdown('</div>', unsafe_allow_html=True)

def render_ai_search_page():
    """
    Render the AI Search page with chat interface.
    """
    st.title(f"{ICONS.get('search', '🤖')} AI Search Assistant")
    st.markdown("Ask questions about your documents and get AI-generated answers based on the content.")
    
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

    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)
            
            # Render sources if available in additional_kwargs
            if isinstance(message, AIMessage):
                kwargs = message.additional_kwargs
                docs = kwargs.get("documents", [])
                needs_clarification = kwargs.get("needs_clarification", False)
                query_analysis = kwargs.get("query_analysis", {})
                
                # Show suggested queries if clarification was needed
                if needs_clarification and query_analysis:
                    suggested = query_analysis.get("suggested_queries", [])
                    if suggested:
                        st.markdown("---")
                        st.markdown("**💡 Try asking:**")
                        cols = st.columns(min(len(suggested), 3))
                        for i, suggestion in enumerate(suggested[:3]):
                            with cols[i]:
                                st.button(
                                    f"📝 {suggestion}", 
                                    key=f"hist_suggest_{idx}_{i}",
                                    on_click=set_suggested_query,
                                    args=(suggestion,),
                                    use_container_width=True
                                )

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
            with st.spinner("Thinking..."):
                try:
                    # Prepare state
                    initial_state = cast(AgentState, {
                        "messages": st.session_state.messages,
                        "question": prompt,
                        "documents": [],
                        "generation": "",
                        "query_analysis": None,
                        "needs_clarification": False,
                        "clarification_response": None
                    })
                    
                    # Run graph
                    response = st.session_state.ai_graph.invoke(initial_state)
                    
                    answer = response.get("generation", "I couldn't generate an answer.")
                    documents = response.get("documents", [])
                    needs_clarification = response.get("needs_clarification", False)
                    query_analysis = response.get("query_analysis", {})
                    
                    message_placeholder.markdown(answer)
                    
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
                            "query_analysis": query_analysis
                        }
                    ))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
