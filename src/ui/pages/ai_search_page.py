import html
import streamlit as st
from typing import cast
from langchain_core.messages import HumanMessage, AIMessage
from src.ai_search.graph import build_graph
from src.ai_search.state import AgentState
from src.ui.theme import ICONS

def render_sources_carousel(documents):
    """
    Render retrieved documents using Streamlit native components.
    """
    if not documents:
        return

    # st.markdown("### Sources")
    
    # Use columns to display documents side-by-side
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
            
            # Determine color based on score
            if score is not None and score >= 0.7:
                color = "#10b981"  # Green for high relevance
            elif score is not None and score >= 0.4:
                color = "#f59e0b"  # Amber for medium relevance
            else:
                color = "#ef4444"  # Red for low relevance
            
            # Use container with border to simulate a card
            with st.container(height=100, border=True):
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 8px;'>
                        <div style='width: 25px; height: 25px; border-radius: 50%; background-color: {color}; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; flex-shrink: 0;'>{score_display}</div>
                        <div style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.85rem;' title='{filename}'>{filename}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(f"<small>{text[:100]}...</small>", unsafe_allow_html=True)

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

    # Display chat messages
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)
            
            # Render sources if available in additional_kwargs
            if isinstance(message, AIMessage) and "documents" in message.additional_kwargs:
                docs = message.additional_kwargs["documents"]
                if docs:
                    render_sources_carousel(docs)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
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
                        "generation": ""
                    })
                    
                    # Run graph
                    response = st.session_state.ai_graph.invoke(initial_state)
                    
                    answer = response.get("generation", "I couldn't generate an answer.")
                    documents = response.get("documents", [])
                    
                    message_placeholder.markdown(answer)
                    
                    # Render sources for the new message
                    if documents:
                        render_sources_carousel(documents)
                    
                    # Add AI message to history with documents
                    st.session_state.messages.append(AIMessage(content=answer, additional_kwargs={"documents": documents}))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
