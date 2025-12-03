from typing import List
from langgraph.graph import StateGraph, END
from src.ai_search.state import AgentState, QueryAnalysis
from src.ai_search.chains import (
    query_rewriter_chain, 
    rag_chain, 
    query_analyzer_chain,
    clarification_chain
)
from src.retrieval.query_engine import QueryEngine
from src.retrieval.similarity_search import SearchResult

# Confidence threshold below which we ask for clarification
CLARITY_THRESHOLD = 0.85


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
    print("---ANALYZE QUERY---")
    question = state["question"]
    messages = state["messages"]
    
    try:
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
        
        return {
            "query_analysis": query_analysis,
            "needs_clarification": needs_clarification
        }
    except Exception as e:
        print(f"Error analyzing query: {e}")
        # On error, proceed without clarification
        return {
            "query_analysis": None,
            "needs_clarification": False
        }


def generate_clarification(state: AgentState):
    """
    Generate a clarification request for the user.
    """
    print("---GENERATE CLARIFICATION---")
    question = state["question"]
    analysis = state.get("query_analysis", {})
    
    clarification = clarification_chain.invoke({
        "question": question,
        "issues": ", ".join(analysis.get("issues", [])),
        "clarifying_questions": ", ".join(analysis.get("clarifying_questions", [])),
        "suggested_queries": ", ".join(analysis.get("suggested_queries", []))
    })
    
    return {
        "generation": clarification,
        "clarification_response": clarification
    }


def rewrite_query(state: AgentState):
    """
    Transform the query to produce a better question.
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    messages = state["messages"]
    
    # If there are no messages (first query), we might not need to rewrite, 
    # but it's often good to normalize it anyway.
    # For now, we always rewrite to ensure it's standalone.
    better_question = query_rewriter_chain.invoke({"messages": messages, "question": question})
    return {"question": better_question}


def retrieve(state: AgentState):
    """
    Retrieve documents based on the query.
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Initialize QueryEngine
    # In a production app, you might want to inject this dependency 
    # or use a singleton to avoid reloading models.
    engine = QueryEngine()
    
    # Perform search
    results = engine.query(query_text=question, top_k=5)
    
    return {"documents": results}


def generate(state: AgentState):
    """
    Generate answer using RAG.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = format_docs(documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    
    return {"generation": generation}


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
