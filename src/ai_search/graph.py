from typing import List
from langgraph.graph import StateGraph, END
from src.ai_search.state import AgentState, QueryAnalysis
from src.ai_search.chains import (
    query_rewriter_chain, 
    rag_chain, 
    query_analyzer_chain,
    clarification_chain
)
from src.ai_search.thinking import ThinkingEmitter, ThinkingUpdate, ThinkingStatus
from src.retrieval.query_engine import QueryEngine
from src.retrieval.similarity_search import SearchResult
from src.utils.config import Config

# Load configuration
config = Config()

# Confidence threshold below which we ask for clarification
CLARITY_THRESHOLD = config.AI_QUERY_CLARITY_THRESHOLD


def format_docs(docs: List[SearchResult]) -> str:
    """Format retrieved documents for the LLM context."""
    formatted = []
    for doc in docs:
        # SearchResult has 'metadata' attribute of type ChunkMetadata
        source = doc.metadata.filename if doc.metadata and hasattr(doc.metadata, 'filename') else "Unknown source"
        formatted.append(f"Source: {source}\nContent: {doc.text}")
    return "\n\n".join(formatted)


def analyze_query(state: AgentState):
    """
    Analyze the query to determine if it's clear enough to answer.
    """
    emitter = ThinkingEmitter("Query Analyzer")
    
    print("---ANALYZE QUERY---")
    question = state["question"]
    messages = state["messages"]
    
    # Emit analyzing update with dynamic status
    thinking_update = emitter.emit_dynamic(
        ThinkingStatus.ANALYZING,
        context=f"Evaluating clarity and intent of: '{question}'",
        progress=0.3
    )
    if "thinking_updates" not in state:
        state["thinking_updates"] = []
    state["thinking_updates"].append(thinking_update)
    
    try:
        # Emit processing update with dynamic status
        processing_update = emitter.emit_dynamic(
            ThinkingStatus.PROCESSING,
            context=f"Extracting semantic information and key concepts from query",
            progress=0.6
        )
        state["thinking_updates"].append(processing_update)
        
        analysis = query_analyzer_chain.invoke({
            "messages": messages, 
            "question": question
        })
        
        # Ensure analysis has all required fields with defaults
        query_analysis: QueryAnalysis = {
            "is_clear": analysis.get("is_clear", True),
            "confidence": analysis.get("confidence", 1.0),
            "issues": analysis.get("issues", []),
            "clarifying_questions": analysis.get("clarifying_questions", []),
            "suggested_queries": analysis.get("suggested_queries", [])
        }
        
        # Determine if clarification is needed
        needs_clarification = (
            not query_analysis["is_clear"] or 
            query_analysis["confidence"] < CLARITY_THRESHOLD
        )
        
        print(f"Query analysis: is_clear={query_analysis['is_clear']}, "
              f"confidence={query_analysis['confidence']}, "
              f"needs_clarification={needs_clarification}")
        
        # Emit complete update with dynamic status
        complete_update = emitter.emit_dynamic(
            ThinkingStatus.COMPLETE,
            context=f"Query analysis finished with {query_analysis['confidence']:.0%} confidence",
            metadata={"is_clear": query_analysis["is_clear"], "confidence": query_analysis["confidence"]}
        )
        state["thinking_updates"].append(complete_update)
        
        return {
            "query_analysis": query_analysis,
            "needs_clarification": needs_clarification,
            "thinking_updates": state["thinking_updates"]
        }
    except Exception as e:
        print(f"Error analyzing query: {e}")
        
        # Emit error update with dynamic status
        error_update = emitter.emit_dynamic(
            ThinkingStatus.ERROR,
            context=f"Query analysis failed: {str(e)}",
        )
        state["thinking_updates"].append(error_update)
        
        # On error, proceed without clarification
        return {
            "query_analysis": None,
            "needs_clarification": False,
            "thinking_updates": state["thinking_updates"]
        }


def generate_clarification(state: AgentState):
    """
    Generate a clarification request for the user.
    """
    emitter = ThinkingEmitter("Clarification Agent")
    
    print("---GENERATE CLARIFICATION---")
    question = state["question"]
    analysis = state.get("query_analysis") or {}
    
    # Emit generating update
    thinking_updates = state.get("thinking_updates", [])
    generating_update = emitter.emit_dynamic(
        ThinkingStatus.GENERATING,
        context="Formulating clarification questions based on query analysis",
        progress=0.4
    )
    thinking_updates.append(generating_update)
    
    # Issues is optional, provide safe default
    issues_list = analysis.get("issues", []) if analysis else []
    questions_list = analysis.get("clarifying_questions", []) if analysis else []
    suggestions_list = analysis.get("suggested_queries", []) if analysis else []
    
    clarification = clarification_chain.invoke({
        "question": question,
        "issues": ", ".join(issues_list) if issues_list else "Query clarity concerns",
        "clarifying_questions": ", ".join(questions_list) if questions_list else "Provide more context",
        "suggested_queries": ", ".join(suggestions_list) if suggestions_list else "More specific query needed"
    })
    
    # Emit complete update
    complete_update = emitter.emit_dynamic(
        ThinkingStatus.COMPLETE,
        context="Clarification prompt ready for user"
    )
    thinking_updates.append(complete_update)
    
    return {
        "generation": clarification,
        "clarification_response": clarification,
        "thinking_updates": thinking_updates
    }


def rewrite_query(state: AgentState):
    """
    Transform the query to produce a better question.
    """
    emitter = ThinkingEmitter("Query Rewriter")
    
    print("---REWRITE QUERY---")
    question = state["question"]
    messages = state["messages"]
    
    thinking_updates = state.get("thinking_updates", [])
    
    # Emit processing update
    processing_update = emitter.emit_dynamic(
        ThinkingStatus.PROCESSING,
        context="Normalizing and optimizing query structure for better retrieval",
        progress=0.5
    )
    thinking_updates.append(processing_update)
    
    # If there are no messages (first query), we might not need to rewrite, 
    # but it's often good to normalize it anyway.
    # For now, we always rewrite to ensure it's standalone.
    better_question = query_rewriter_chain.invoke({"messages": messages, "question": question})
    
    # Emit complete update
    complete_update = emitter.emit_dynamic(
        ThinkingStatus.COMPLETE,
        context="Query optimization complete and ready for knowledge base search"
    )
    thinking_updates.append(complete_update)
    
    return {
        "question": better_question,
        "thinking_updates": thinking_updates
    }


def retrieve(state: AgentState):
    """
    Retrieve documents based on the query.
    """
    emitter = ThinkingEmitter("Document Retriever")
    
    print("---RETRIEVE---")
    question = state["question"]
    
    thinking_updates = state.get("thinking_updates", [])
    
    # Emit retrieving update
    retrieving_update = emitter.emit_dynamic(
        ThinkingStatus.RETRIEVING,
        context=f"Searching knowledge base for documents related to: {question[:50]}...",
        progress=0.3
    )
    thinking_updates.append(retrieving_update)

    # Initialize QueryEngine
    # In a production app, you might want to inject this dependency 
    # or use a singleton to avoid reloading models.
    engine = QueryEngine()
    
    # Emit processing update
    processing_update = emitter.emit_dynamic(
        ThinkingStatus.PROCESSING,
        context="Computing semantic similarities and ranking documents by relevance",
        progress=0.7
    )
    thinking_updates.append(processing_update)
    
    # Perform search
    results = engine.query(query_text=question, top_k=5)
    
    # Emit complete update
    complete_update = emitter.emit_dynamic(
        ThinkingStatus.COMPLETE,
        context=f"Document search complete with {len(results)} relevant sources found",
        metadata={"documents_count": len(results)}
    )
    thinking_updates.append(complete_update)
    
    return {
        "documents": results,
        "thinking_updates": thinking_updates
    }


def generate(state: AgentState):
    """
    Generate answer using RAG.
    """
    emitter = ThinkingEmitter("Answer Generator")
    
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    thinking_updates = state.get("thinking_updates", [])
    
    # Emit reasoning update
    reasoning_update = emitter.emit_dynamic(
        ThinkingStatus.REASONING,
        context="Analyzing retrieved documents and evaluating relevance for the query",
        progress=0.3
    )
    thinking_updates.append(reasoning_update)
    
    context = format_docs(documents)
    
    # Emit generating update
    generating_update = emitter.emit_dynamic(
        ThinkingStatus.GENERATING,
        context="Synthesizing information from sources to create comprehensive answer",
        progress=0.7
    )
    thinking_updates.append(generating_update)
    
    generation = rag_chain.invoke({"context": context, "question": question})
    
    # Emit complete update
    complete_update = emitter.emit_dynamic(
        ThinkingStatus.COMPLETE,
        context="Answer generation complete and ready to present",
        metadata={"sources_used": len(documents)}
    )
    thinking_updates.append(complete_update)
    
    return {
        "generation": generation,
        "thinking_updates": thinking_updates
    }


def route_after_analysis(state: AgentState) -> str:
    """
    Route to clarification or continue with retrieval based on query analysis.
    """
    if state.get("needs_clarification", False):
        return "generate_clarification"
    return "rewrite_query"


def build_graph():
    """
    Build and compile the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("generate_clarification", generate_clarification)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Define edges with conditional routing
    workflow.set_entry_point("analyze_query")
    
    # After analysis, decide whether to ask for clarification or proceed
    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "generate_clarification": "generate_clarification",
            "rewrite_query": "rewrite_query"
        }
    )
    
    # Clarification ends the flow (user needs to respond)
    workflow.add_edge("generate_clarification", END)
    
    # Normal flow continues
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app
