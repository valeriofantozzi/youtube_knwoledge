from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from src.retrieval.similarity_search import SearchResult


class QueryAnalysis(TypedDict):
    """
    Result of analyzing a user query for clarity and specificity.
    """
    is_clear: bool  # Whether the query is clear enough to answer
    confidence: float  # Confidence score (0-1)
    issues: List[str]  # List of issues with the query
    clarifying_questions: List[str]  # Suggested clarifying questions
    suggested_queries: List[str]  # Better formulated query suggestions


class AgentState(TypedDict):
    """
    State for the RAG agent graph.
    """
    messages: List[BaseMessage]
    question: str
    documents: List[SearchResult]
    generation: str
    # New fields for smart query handling
    query_analysis: Optional[QueryAnalysis]
    needs_clarification: bool
    clarification_response: Optional[str]
