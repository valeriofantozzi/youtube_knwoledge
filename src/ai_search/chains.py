from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.utils.config import Config
from src.ai_search.llm_factory import LLMFactory

# Initialize Config
config = Config()

# Initialize Factory
LLMFactory.initialize(config)

# Initialize LLMs for each agent with their specific configuration
llm_query_analyzer = LLMFactory.create_query_analyzer_llm()
llm_clarification = LLMFactory.create_clarification_llm()
llm_query_rewriter = LLMFactory.create_query_rewriter_llm()
llm_rag_generator = LLMFactory.create_rag_generator_llm()

# --- Query Analyzer Chain ---
query_analyzer_system_prompt = """You are an expert query analyzer. Your task is to evaluate if a user's question is clear and specific enough to search a knowledge base effectively.

Analyze the question and determine:
1. Is the question clear and specific enough to answer? (is_clear: true/false)
2. How confident are you in this assessment? (confidence: 0.0-1.0)
3. What issues exist with the question, if any? (issues: list of strings)
4. What clarifying questions should be asked to improve the query? (clarifying_questions: list of strings)
5. What better-formulated versions of the query would you suggest? (suggested_queries: list of strings)

A question is considered UNCLEAR or VAGUE if:
- It's too broad (e.g., "tell me about orchids" vs "how often should I water orchids?")
- It asks for a general process without specific details (e.g. "How to care for orchids?" is vague because it depends on the type of orchid, environment, etc. Better: "How to water Phalaenopsis orchids?")
- It lacks context (e.g., "how to fix it?" without specifying what)
- It's ambiguous (multiple interpretations possible)
- It's a single word or very short without clear intent

A question is CLEAR if:
- It has a specific topic and intent
- It can be answered with concrete information
- The user's need is evident

IMPORTANT: Respond ONLY with valid JSON, no additional text.

Example output for a vague question:
{{
    "is_clear": false,
    "confidence": 0.85,
    "issues": ["The question is too broad", "No specific aspect mentioned"],
    "clarifying_questions": ["What specific aspect interests you?", "Are you looking for care tips, varieties, or something else?"],
    "suggested_queries": ["How to care for orchids?", "What are the most common orchid varieties?", "How often should orchids be watered?"]
}}

Example output for a clear question:
{{
    "is_clear": true,
    "confidence": 0.95,
    "issues": [],
    "clarifying_questions": [],
    "suggested_queries": []
}}
"""

query_analyzer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_analyzer_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Analyze this question: {question}"),
    ]
)

query_analyzer_chain = query_analyzer_prompt | llm_query_analyzer | JsonOutputParser()


# --- Query Rewriter Chain ---
rephrase_system_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the question is already standalone, return it as is.
"""

rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrase_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}"),
    ]
)

query_rewriter_chain = rephrase_prompt | llm_query_rewriter | StrOutputParser()


# --- Clarification Response Generator Chain ---
clarification_system_prompt = """You are a helpful assistant. The user asked a question that is too vague or unclear to answer effectively.

Based on the analysis provided, generate a friendly response that:
1. Acknowledges their question
2. Explains briefly why you need more information
3. Asks clarifying questions to help them get a better answer
4. Suggests some specific questions they might want to ask instead

Be conversational and helpful, not robotic. Keep the response concise.

Analysis of the question:
- Issues identified: {issues}
- Clarifying questions: {clarifying_questions}
- Suggested better queries: {suggested_queries}

Respond in the same language as the user's question.
"""

clarification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", clarification_system_prompt),
        ("human", "Original question: {question}"),
    ]
)

clarification_chain = clarification_prompt | llm_clarification | StrOutputParser()


# --- RAG Generator Chain ---
rag_system_prompt = """You are an assistant for question-answering tasks. 
Answer the question based ONLY on the following pieces of retrieved context. 
If the answer is not in the context, say that you cannot answer based on the available information.
Do NOT use your internal knowledge.
Use three sentences maximum and keep the answer concise.

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        ("human", "{question}"),
    ]
)

rag_chain = rag_prompt | llm_rag_generator | StrOutputParser()


# --- Thinking Status Generator Chains ---
# These chains generate dynamic thinking status messages based on what the agent is doing

# --- Thinking Status Generator Chains ---
# These chains generate dynamic thinking status messages based on what the agent is doing

thinking_status_system_prompt = """You are a cheerful, witty thinking process narrator with personality!
Your job is to generate ONE fun, action-oriented sentence (4-7 words max) describing what's happening RIGHT NOW.
Include ONE relevant emoji at the START of the sentence.

CRITICAL RULES:
1. Be SPECIFIC, action-oriented, and FUN (not generic or boring)
2. Use PRESENT PROGRESSIVE verbs (ing form): analyzing, searching, processing, evaluating, extracting, building, synthesizing, examining, decoding, hunting, etc.
3. Match the context's domain and operation
4. NEVER repeat the same sentence twice
5. Be concise, natural, and playful sounding
6. START with emoji, then space, then text
7. ONLY output the sentence, no punctuation at end, no extra text
8. Use personality: "Let's", "hunting for", "diving into", "untangling", "cooking up", "hunting down", etc.

EXAMPLES by operation type:

ANALYZING QUERY:
- "🔎 Decoding your mysterious query intent"
- "🧠 Untangling semantic meaning patterns"
- "🎯 Pinpointing key concepts and domains"
- "🔬 Examining query structure carefully"
- "💭 Parsing what you really mean"

EXTRACTING/PROCESSING:
- "⚡ Distilling semantic gold from noise"
- "🔗 Mapping concept relationships smoothly"
- "📊 Tokenizing language features cleverly"
- "🧬 Extracting domain terminology patterns"
- "🎨 Sculpting semantic features"

SEARCHING/RETRIEVING:
- "🏃 Hunting through knowledge base fast"
- "📚 Surfing through document indices"
- "🎲 Filtering semantic matches perfectly"
- "🔀 Ranking documents by brilliance"
- "🧲 Aggregating source materials"

GENERATING/SYNTHESIZING:
- "🎬 Weaving sources into magic"
- "🧩 Composing evidence-backed wisdom"
- "🏗️ Building information architecture"
- "🌊 Integrating knowledge streams smoothly"
- "✨ Brewing response carefully"
- "🎭 Orchestrating answer symphony"
- "🍳 Cooking up brilliant insights"

Now, based on the context, generate a single unique status sentence with emoji:"""

thinking_status_prompt = ChatPromptTemplate.from_template(
    thinking_status_system_prompt + "\n\nContext: {context}\n\nGenerate the status:"
)

# Create a thinking status generator that uses a simple lightweight model
thinking_status_generator = thinking_status_prompt | llm_query_analyzer | StrOutputParser()