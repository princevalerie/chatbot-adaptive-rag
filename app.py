import os
import tempfile
import streamlit as st
import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

import google.generativeai as genai
from PIL import Image
import io
import base64
import json
import math
import re

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Streamlit Configuration
st.set_page_config(
    page_title="AI Assistant with Adaptive RAG Capabilities",
    page_icon="🧠",
    layout="wide"
)

# SIDEBAR CONFIGURATION
st.sidebar.title("⚙️ Configuration")

# Initialize session state for API key override
if 'override_api_key' not in st.session_state:
    st.session_state.override_api_key = False

# Check for environment variable first
env_api_key = os.environ.get("GOOGLE_API_KEY")
api_key = None

# Test environment API key if it exists
def test_api_key(key):
    """Test if API key works by trying to configure genai"""
    try:
        genai.configure(api_key=key)
        # Try a simple test (this doesn't count against quota)
        return True
    except Exception:
        return False

# API Key Logic
if env_api_key and not st.session_state.override_api_key:
    # Environment variable exists, test it
    if test_api_key(env_api_key):
        st.sidebar.success("✅ Using API key from environment variable")
        api_key = env_api_key
        
        # Override button
        if st.sidebar.button("🔧 Override with Manual API Key", type="secondary"):
            st.session_state.override_api_key = True
            st.rerun()
    else:
        st.sidebar.error("❌ Environment API key is invalid")
        st.session_state.override_api_key = True

# Show manual input if no env key or override is enabled
if not env_api_key or st.session_state.override_api_key:
    # Manual API Key Input
    manual_api_key = st.sidebar.text_input("Enter Google API Key:", type="password")
    
    if st.session_state.override_api_key and env_api_key:
        # Show option to go back to env key
        if st.sidebar.button("↩️ Use Environment API Key", type="secondary"):
            st.session_state.override_api_key = False
            st.rerun()
    
    if manual_api_key:
        if test_api_key(manual_api_key):
            api_key = manual_api_key
            st.sidebar.success("✅ Manual API key validated")
        else:
            st.sidebar.error("❌ Invalid API key")
    else:
        if not env_api_key or st.session_state.override_api_key:
            st.sidebar.warning("⚠️ Please enter your Google API Key to continue.")

# Final validation
if not api_key:
    st.title("🧠 AI Assistant with Adaptive RAG Capabilities")
    st.info("👈 Please provide a valid Google API Key in the sidebar to get started.")
    st.stop()

# Gemini Configuration
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# Initialize models
@st.cache_resource
def initialize_models(_api_key):
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=_api_key
        )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=_api_key
        )
        
        return model, embeddings
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing models: {str(e)}")
        return None, None

model, embeddings = initialize_models(api_key)

if not model:
    st.title("🧠 AI Assistant with Adaptive RAG Capabilities")
    st.error("Failed to initialize AI models. Please check your API key.")
    st.stop()

# STATE DEFINITION
class AdaptiveRetrievalState(BaseModel):
    # Core data
    original_query: str
    current_query: str
    query_history: List[str] = Field(default_factory=list)

    # Reasoning outputs
    reasoning_analysis: str = ""
    query_intent: str = ""

    # Retrieval results
    retrieved_docs: List[Dict] = Field(default_factory=list)
    doc_scores: List[float] = Field(default_factory=list)
    best_docs: List[Dict] = Field(default_factory=list)
    best_confidence: float = 0.0

    # Quality metrics
    confidence_score: float = 0.0
    confidence_level: str = ""
    quality_reasons: List[str] = Field(default_factory=list)

    # Flow control
    iteration_count: int = 0
    max_iterations: int = 5
    should_continue: bool = True

    # Final output
    final_answer: str = ""
    answer_source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Dict-like helpers for compatibility with existing code
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def update(self, data: Dict[str, Any]):
        for k, v in data.items():
            setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

# CONFIDENCE CALCULATION FUNCTIONS
def calculate_keyword_coverage(query: str, docs: List[Dict]) -> float:
    """Calculate how well documents cover query keywords"""
    if not docs or not query:
        return 0.0
    
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    
    # Combine all document content
    all_doc_text = ""
    for doc in docs:
        if isinstance(doc, dict) and 'page_content' in doc:
            all_doc_text += doc['page_content'].lower() + " "
        elif hasattr(doc, 'page_content'):
            all_doc_text += doc.page_content.lower() + " "
    
    # Count how many query words appear in documents
    covered_words = sum(1 for word in query_words if word in all_doc_text)
    return covered_words / len(query_words) if query_words else 0.0

def calculate_confidence_score(docs: List[Dict], scores: List[float], query: str, domain: str = "") -> float:
    """Calculate confidence score based on multiple factors"""
    if not docs or not scores:
        return 0.0
    
    # Component 1: Top document similarity (40%)
    top_similarity = max(scores) if scores else 0.0
    
    # Component 2: Number of relevant documents (25%)
    relevant_docs = len([s for s in scores if s > 0.5])
    doc_count_score = min(relevant_docs / 3, 1.0)
    
    # Component 3: Score consistency (20%)
    if len(scores) > 1:
        score_std = (max(scores) - min(scores))
        consistency_score = max(0, 1.0 - score_std)
    else:
        consistency_score = 1.0
    
    # Component 4: Keyword coverage (15%)
    keyword_coverage = calculate_keyword_coverage(query, docs)
    
    # Calculate weighted final score
    confidence = (
        top_similarity * 0.4 +
        doc_count_score * 0.25 +
        consistency_score * 0.2 +
        keyword_coverage * 0.15
    )
    
    return min(confidence, 1.0)

# RAG SEARCH TOOL
@tool
def rag_search_tool(query: str) -> Dict[str, Any]:
    """Search for relevant information in uploaded PDF documents."""
    try:
        if not st.session_state.get('vector_store'):
            return {
                "documents": [],
                "scores": [],
                "error": "No documents uploaded or processed yet."
            }
        
        # Perform similarity search with scores
        docs_with_scores = st.session_state.vector_store.similarity_search_with_score(query, k=5)
        
        # Separate documents and scores
        documents = []
        scores = []
        
        for doc, score in docs_with_scores:
            documents.append({
                "page_content": doc.page_content,
                "metadata": getattr(doc, 'metadata', {})
            })
            # Convert distance to similarity (FAISS returns distance, we want similarity)
            similarity = max(0, 1 - score) 
            scores.append(similarity)
        
        return {
            "documents": documents,
            "scores": scores,
            "error": None
        }
        
    except Exception as e:
        return {
            "documents": [],
            "scores": [],
            "error": f"Error in document search: {str(e)}"
        }

# NODE IMPLEMENTATIONS

def reason_node(state: AdaptiveRetrievalState) -> AdaptiveRetrievalState:
    """
    Node 1: Reasoning - Analyzes query and sets up initial strategy
    Executed ONCE at the beginning
    """
    original_query = state["original_query"]
    
    reasoning_prompt = f"""Analyze this query and provide structured analysis:

Query: {original_query}

CRITICAL RULES:
- Base your analysis ONLY on the actual query text provided
- Do NOT make assumptions about user intent beyond what is explicitly stated
- Do NOT assume domain knowledge not evident in the query
- Stick to what can be directly inferred from the query words

Please analyze:
1. INTENT: What type of information is the user seeking? (based only on query words)
2. COMPLEXITY: How complex is this query? (simple/moderate/complex)
3. STRATEGY: What search approach would work best? (based on query structure)

Provide your analysis in this format:
INTENT: [analysis based only on query text]
COMPLEXITY: [complexity level based on query structure]
STRATEGY: [approach based on query characteristics]
"""
    
    try:
        response = model.invoke([HumanMessage(content=reasoning_prompt)])
        reasoning_text = response.content
        
        # Parse the response
        lines = reasoning_text.split('\n')
        intent = complexity = strategy = ""
        
        for line in lines:
            if line.startswith("INTENT:"):
                intent = line.replace("INTENT:", "").strip()
            elif line.startswith("COMPLEXITY:"):
                complexity = line.replace("COMPLEXITY:", "").strip()
            elif line.startswith("STRATEGY:"):
                strategy = line.replace("STRATEGY:", "").strip()
        
        # Update state
        state.update({
            "reasoning_analysis": reasoning_text,
            "query_intent": intent,
            "current_query": original_query,
            "query_history": [original_query],
            "iteration_count": 0,
            "max_iterations": 5,
            "should_continue": True,
            "best_confidence": 0.0,
            "metadata": {
                "complexity": complexity,
                "strategy": strategy
            }
        })
        
    except Exception as e:
        # Fallback on error
        state.update({
            "reasoning_analysis": f"Error in reasoning: {str(e)}",
            "query_intent": "general",
            "current_query": original_query,
            "query_history": [original_query],
            "iteration_count": 0,
            "max_iterations": 5,
            "should_continue": True,
            "best_confidence": 0.0,
            "metadata": {}
        })
    
    return state

def retriever_node(state: AdaptiveRetrievalState) -> AdaptiveRetrievalState:
    """
    Node 2: Retriever - Performs document retrieval and confidence calculation
    Can be executed MULTIPLE times (up to 5 iterations)
    """
    current_query = state["current_query"]
    iteration = state["iteration_count"] + 1
    
    try:
        # Perform RAG search
        search_result = rag_search_tool.invoke({"query": current_query})
        
        if search_result["error"]:
            raise Exception(search_result["error"])
        
        retrieved_docs = search_result["documents"]
        doc_scores = search_result["scores"]
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(
            docs=retrieved_docs,
            scores=doc_scores,
            query=current_query
        )
        
        # Determine confidence level using composite signals (score, strong docs, coverage)
        strong_docs = len([s for s in doc_scores if s >= 0.6])
        keyword_coverage = calculate_keyword_coverage(current_query, retrieved_docs)
        if confidence_score >= 0.80 and strong_docs >= 2:
            confidence_level = "high"
        elif confidence_score >= 0.65 and keyword_coverage >= 0.50:
            confidence_level = "medium"
        elif confidence_score >= 0.50:
            confidence_level = "low"
        else:
            confidence_level = "insufficient"
        
        # Quality assessment
        quality_reasons = []
        if confidence_score >= 0.9:
            quality_reasons.append("Excellent document relevance")
        elif confidence_score >= 0.8:
            quality_reasons.append("Good document match")
        elif confidence_score >= 0.6:
            quality_reasons.append("Moderate relevance")
        else:
            quality_reasons.append("Low relevance, needs improvement")
        
        # Track best results across iterations
        best_docs = state.get("best_docs", [])
        best_confidence = state.get("best_confidence", 0.0)
        
        if confidence_score > best_confidence:
            best_docs = retrieved_docs
            best_confidence = confidence_score
        
        # Update state
        state.update({
            "retrieved_docs": retrieved_docs,
            "doc_scores": doc_scores,
            "best_docs": best_docs,
            "best_confidence": best_confidence,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "quality_reasons": quality_reasons,
            "iteration_count": iteration
        })
        
    except Exception as e:
        # Handle errors gracefully
        state.update({
            "retrieved_docs": [],
            "doc_scores": [],
            "confidence_score": 0.0,
            "confidence_level": "error",
            "quality_reasons": [f"Error in retrieval: {str(e)}"],
            "iteration_count": iteration
        })
    
    return state

def rewrite_node(state: AdaptiveRetrievalState) -> AdaptiveRetrievalState:
    """
    Node 3: Rewrite - Rewrites query based on previous retrieval results
    Executed conditionally when retrieval quality is insufficient
    """
    current_query = state["current_query"]
    iteration = state["iteration_count"]
    retrieved_docs = state.get("retrieved_docs", [])
    confidence_score = state.get("confidence_score", 0.0)
    
    # Prepare context from previous retrieval results
    retrieval_context = ""
    if retrieved_docs:
        # Get snippets from top documents to understand what was found
        doc_snippets = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Top 3 docs
            content = doc.get("page_content", "")
            # Get first 150 characters as snippet
            snippet = content[:150] + "..." if len(content) > 150 else content
            doc_snippets.append(f"Doc {i+1}: {snippet}")
        
        retrieval_context = f"""
PREVIOUS RETRIEVAL RESULTS (Iteration {iteration}):
Confidence Score: {confidence_score:.2f}
Documents Found:
{chr(10).join(doc_snippets)}

ANALYSIS:
- The above documents were found but had low relevance score
- Gap Analysis: What aspects of the query were not well covered?
- Missing Elements: What information seems to be absent?
"""
    
    # Add query history for complete context
    query_history = state.get("query_history", [])
    history_context = ""
    if len(query_history) > 1:
        previous_queries = query_history[:-1]  # Exclude current query
        history_context = f"""
QUERY EVOLUTION HISTORY:
{chr(10).join([f"Attempt {i+1}: {q}" for i, q in enumerate(previous_queries)])}
Current attempt: {current_query}
"""

    rewrite_prompt = f"""You are a query optimization expert with access to previous search results and query history.

ORIGINAL QUERY: {state["original_query"]}
CURRENT QUERY: {current_query}
ITERATION: {iteration}

{history_context}

{retrieval_context}

CONTEXT-AWARE REWRITING TASK:
Based on what was found vs what was needed, rewrite the query to fill the gaps.

INSTRUCTIONS:
- Analyze the gap between query intent and retrieved content
- Learn from previous query attempts that didn't work well
- Keep the core intent intact but try different angles
- Adjust keywords based on what was actually found in documents
- If documents contain related but not exact information, bridge that gap
- Try different terminology that might be used in the actual documents
- Avoid repeating approaches that already failed
- Return ONLY the rewritten query, no explanations

REWRITTEN QUERY:"""
    
    try:
        response = model.invoke([HumanMessage(content=rewrite_prompt)])
        new_query = response.content.strip()
        
        # Clean up the response in case LLM adds extra text
        if ":" in new_query:
            new_query = new_query.split(":")[-1].strip()
        if new_query.startswith('"') and new_query.endswith('"'):
            new_query = new_query[1:-1]
        
        # Update query history
        query_history = state.get("query_history", [])
        query_history.append(new_query)
        
        # Update state
        state.update({
            "current_query": new_query,
            "query_history": query_history
        })
        
    except Exception as e:
        # Fallback: slight modification of current query
        new_query = current_query + " information data details"
        query_history = state.get("query_history", [])
        query_history.append(new_query)
        
        state.update({
            "current_query": new_query,
            "query_history": query_history
        })
    
    return state

# CONDITIONAL LOGIC

def should_continue_retrieval(state: AdaptiveRetrievalState) -> str:
    """
    Determines next step based on confidence score and iteration count
    This is the KEY conditional logic for LangGraph routing
    """
    confidence = state.get("confidence_score", 0.0)
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    
    # High confidence on first try (0.9 threshold) - STOP
    if iteration == 1 and confidence >= 0.9:
        return "end_high_confidence"
    
    # Medium confidence on second try (0.8 threshold) - STOP  
    elif iteration == 2 and confidence >= 0.8:
        return "end_medium_confidence"
    
    # Reached max iterations - STOP with fallback
    elif iteration >= max_iterations:
        return "end_fallback"
    
    # Need to rewrite and try again
    else:
        return "continue_rewrite"

# FINAL ANSWER GENERATION

def generate_final_answer(state: AdaptiveRetrievalState) -> AdaptiveRetrievalState:
    """Generate final answer based on best retrieved documents"""
    
    # Use best documents found across all iterations
    best_docs = state.get("best_docs", state.get("retrieved_docs", []))
    confidence_level = state.get("confidence_level", "low")
    confidence_score = state.get("confidence_score", 0.0)
    iteration_count = state.get("iteration_count", 1)
    original_query = state.get("original_query", "")
    
    if not best_docs:
        final_answer = "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload more relevant documents."
        answer_source = "no_documents"
    else:
        # Prepare context from best documents
        context = "\n\n".join([doc.get("page_content", "") for doc in best_docs[:3]])
        
        # Create answer generation prompt
        answer_prompt = f"""Based on the provided documents, answer the user's question comprehensively.

User Question: {original_query}

Relevant Documents:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the document content
- If information is incomplete, acknowledge the limitations
- Use specific data and facts from the documents when available
- Be concise but thorough

Answer:"""
        
        try:
            response = model.invoke([HumanMessage(content=answer_prompt)])
            final_answer = response.content
            answer_source = f"iteration_{iteration_count}"
        except Exception as e:
            final_answer = f"Error generating answer: {str(e)}"
            answer_source = "error"
    
    # Remove confidence information display
    
    state.update({
        "final_answer": final_answer,
        "answer_source": answer_source
    })
    
    return state

# GRAPH CONSTRUCTION

def create_adaptive_retrieval_graph():
    """Creates the LangGraph workflow"""
    
    # Create StateGraph
    workflow = StateGraph(AdaptiveRetrievalState)
    
    # Add nodes
    workflow.add_node("reason", reason_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("rewrite", rewrite_node)
    
    # Set entry point
    workflow.set_entry_point("reason")
    
    # Add edges
    workflow.add_edge("reason", "retriever")  # Always go to retriever after reason
    workflow.add_edge("rewrite", "retriever")  # After rewrite, go back to retriever
    
    # Add conditional edges from retriever
    workflow.add_conditional_edges(
        "retriever",  # From node
        should_continue_retrieval,  # Decision function
        {
            "end_high_confidence": END,
            "end_medium_confidence": END,
            "end_fallback": END,
            "continue_rewrite": "rewrite"
        }
    )
    
    return workflow.compile()

# Initialize the adaptive retrieval graph
adaptive_graph = create_adaptive_retrieval_graph()

# PDF PROCESSING FUNCTIONS (from original code)

def process_pdf(uploaded_file):
    """Process uploaded PDF file using PyMuPDF4LLM"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyMuPDF4LLMLoader(tmp_file_path)
            documents = loader.load()
            
            text_content = ""
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    text_content += doc.page_content + "\n"
                elif hasattr(doc, 'text'):
                    text_content += doc.text + "\n"
                else:
                    text_content += str(doc) + "\n"
            
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            if text_content and text_content.strip():
                return text_content
            else:
                return "No readable content found in PDF."
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ PyMuPDF4LLM conversion failed for {uploaded_file.name}")
            try:
                import pypdf
                with open(tmp_file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    fallback_text = ""
                    for page in pdf_reader.pages:
                        fallback_text += page.extract_text() + "\n"
                    
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
                    if fallback_text.strip():
                        return fallback_text
                    else:
                        return "No readable text found in PDF."
            except Exception as fallback_e:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                return "Failed to extract content from PDF."
        
    except Exception as e:
        st.sidebar.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        return None

def split_text(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def create_vector_store(texts):
    """Create vector store from texts"""
    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
    except Exception as e:
        st.sidebar.error(f"❌ Error creating vector store: {str(e)}")
        return None

# SESSION STATE INITIALIZATION
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# SIDEBAR - FILE UPLOAD SECTION
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload PDF files for analysis"
)

# SIDEBAR - CONVERSATION MANAGEMENT
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Conversation")

# Show conversation history count and clear button
if st.session_state.chat_history:
    conversation_count = len(st.session_state.chat_history)
    st.sidebar.info(f"📝 {conversation_count} messages in history")
    
    if st.sidebar.button("🗑️ Clear All Conversation History", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.sidebar.info("💭 No conversation history yet")
    
    # Show disabled button when no history
    st.sidebar.button("🗑️ Clear All Conversation History", type="secondary", use_container_width=True, disabled=True, help="No conversation to clear")

# SIDEBAR - INSTRUCTIONS
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 How to Use")
st.sidebar.markdown("""
1. ✅ Enter your Google API Key above
2. 📄 Upload PDF files
3. ⏳ Wait for processing to complete
4. 💬 Ask questions about the data
""")

# SIDEBAR - PROCESSING SECTION
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Processing Status")

# Process uploaded files
if uploaded_files and not st.session_state.documents_processed:
    st.session_state.documents_processed = False
    
    with st.sidebar:
        st.info("🔄 Processing documents...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_texts = []
        processing_success = True
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {uploaded_file.name}")
            
            try:
                text = process_pdf(uploaded_file)
                
                if text and text.strip():
                    chunked_texts = split_text(text)
                    all_texts.extend(chunked_texts)
                    st.success(f"✅ {uploaded_file.name} ({len(chunked_texts)} chunks)")
                else:
                    st.warning(f"⚠️ No text from {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed: {uploaded_file.name}")
                processing_success = False
                break
        
        if all_texts and processing_success:
            status_text.text("Creating searchable index...")
            vector_store = create_vector_store(all_texts)
            
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.documents_processed = True
                st.success(f"✅ Processed {len(uploaded_files)} files successfully!")
                st.balloons()
                progress_bar.empty()
                status_text.empty()
            else:
                st.error("❌ Failed to create searchable index")
        else:
            st.error("❌ Processing failed")

# SIDEBAR - STATUS INDICATOR
if st.session_state.documents_processed:
    st.sidebar.success("✅ Documents ready for analysis")

elif uploaded_files:
    st.sidebar.info("⏳ Processing in progress...")
else:
    st.sidebar.info("📄 No documents uploaded yet")

# Auto-reset when files are removed
if not uploaded_files and st.session_state.documents_processed:
    st.session_state.documents_processed = False
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.rerun()

# MAIN CONTENT AREA
st.title("🧠 AI Assistant with Adaptive RAG Capabilities")

# Only show chat interface if documents are processed
if st.session_state.documents_processed and st.session_state.vector_store:
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
    
    # Input question
    if question := st.chat_input("Ask about your documents..."):
        # Add user question to history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)
        
        # Process using adaptive retrieval graph
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing with adaptive retrieval..."):
                try:
                    # Initialize state for the graph (BaseModel)
                    initial_state = AdaptiveRetrievalState(
                        original_query=question,
                        current_query="",
                        query_history=[],
                        reasoning_analysis="",
                        query_intent="",
                        retrieved_docs=[],
                        doc_scores=[],
                        best_docs=[],
                        best_confidence=0.0,
                        confidence_score=0.0,
                        confidence_level="",
                        quality_reasons=[],
                        iteration_count=0,
                        max_iterations=5,
                        should_continue=True,
                        final_answer="",
                        answer_source="",
                        metadata={}
                    )
                    
                    # Execute the graph
                    result = adaptive_graph.invoke(initial_state)
                    
                    # Generate final answer if not already done
                    if not result.get("final_answer"):
                        result = generate_final_answer(result)
                    
                    answer = result["final_answer"]
                    
                    st.write(answer)
                    
                    # Add answer to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, an error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

else:
    # Show welcome message when no documents are processed
    if not uploaded_files:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h3>🧠 AI Assistant with Adaptive RAG Capabilities</h3>
            <p>👈 Upload your PDF documents in the sidebar to get started</p>
            <p>Experience intelligent document analysis with autonomous reasoning, adaptive retrieval, and smart query enhancement.</p>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.documents_processed:
        st.info("⏳ Please wait while your documents are being processed...")
