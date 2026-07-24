import os
import sys
import time
import streamlit as st

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch  # Critical: Import torch first to prevent shm.dll load conflicts with PaddlePaddle
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.intent_clarification.clarifier import IntentClarifier

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="SmartLogi - FAQ & Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Body Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Glassmorphism containers */
    .source-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    .source-title {
        font-weight: 600;
        color: #4D96FF;
        margin-bottom: 5px;
        font-size: 0.95rem;
    }
    
    .source-meta {
        font-size: 0.8rem;
        color: #718096;
    }
    
    /* Custom chat bubbles */
    .user-bubble {
        background-color: #2D3748;
        border-radius: 15px 15px 0px 15px;
        padding: 15px;
        margin-bottom: 10px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 15px 15px 15px 0px;
        padding: 15px;
        margin-bottom: 10px;
        color: #F7FAFC;
        border: 1px solid rgba(77, 150, 255, 0.2);
    }

    /* Option button styles */
    .clarification-option {
        background-color: #2D3748;
        color: white;
        border-radius: 8px;
        border: 1px solid #4D96FF;
        padding: 10px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .clarification-option:hover {
        background-color: #4D96FF;
        color: white;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Lazy Initialization of Backend Components using session_state
# -----------------------------------------------------------------------------
if "embedder" not in st.session_state:
    with st.spinner("Loading embedding models..."):
        st.session_state.embedder = Embedder()
        st.session_state.db_manager = VectorDBManager(embedder=st.session_state.embedder)
        st.session_state.retriever = HybridRetriever(
            db_manager=st.session_state.db_manager, 
            embedder=st.session_state.embedder
        )
        st.session_state.llm_client = LLMClient()
        st.session_state.pipeline = RAGPipeline(
            retriever=st.session_state.retriever, 
            llm_client=st.session_state.llm_client
        )
        st.session_state.clarifier = IntentClarifier(llm_client=st.session_state.llm_client)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
    st.session_state.clarification_options = []
    st.session_state.original_query = ""

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Pipeline Parameters")

faq_threshold = st.sidebar.slider(
    "FAQ Match Threshold", 
    min_value=0.50, 
    max_value=0.95, 
    value=0.70, 
    step=0.05,
    help="Cosine similarity score required to return an FAQ directly. Lower value = more FAQ direct hits."
)

top_k = st.sidebar.slider(
    "Document Chunks (Top-K)", 
    min_value=1, 
    max_value=8, 
    value=4,
    help="Number of document chunks to retrieve if FAQ does not match."
)

temperature = st.sidebar.slider(
    "LLM Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.2, 
    step=0.1,
    help="Higher values make generation more creative, lower values make it more focused and grounded."
)

st.sidebar.markdown("---")
# Health Check Display in Sidebar
if st.session_state.llm_client.health_check():
    st.sidebar.success("🟢 LM Studio Server: ONLINE")
else:
    st.sidebar.error("🔴 LM Studio Server: OFFLINE")
    st.sidebar.info("Please start the LM Studio local server on port 1234.")

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>SmartLogi Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Retrieve project specs, deployment guides, and FAQs securely using RAG with Intent Clarification</div>", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if assistant message has them
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 View References"):
                st.markdown(f"**Retrieval Source**: `{msg['retrieved_from'].upper()}`")
                for i, src in enumerate(msg["sources"]):
                    if src.get("type") == "faq":
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<div class='source-title'>[{i+1}] FAQ: {src.get('source')}</div>"
                            f"<div class='source-meta'>Category: {src.get('category')} | Cosine Similarity: {src.get('score'):.4f}</div>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<div class='source-title'>[{i+1}] File: {src.get('source')} (Chunk {src.get('chunk_index')+1}/{src.get('total_chunks')})</div>"
                            f"<div class='source-meta'>Category: {src.get('category')} | Cosine Similarity: {src.get('score'):.4f}</div>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )

# -----------------------------------------------------------------------------
# Intent Clarification Option Buttons
# -----------------------------------------------------------------------------
if st.session_state.awaiting_clarification:
    st.info("⚠️ I found multiple possible meanings. Please select one option below:")
    
    # Render buttons for each option
    for idx, opt in enumerate(st.session_state.clarification_options):
        if st.button(f"Option {idx+1}: {opt}", key=f"opt_{idx}"):
            # User selected an option, clear state and run pipeline
            selected_query = opt
            st.session_state.awaiting_clarification = False
            
            # Display user's selection in chat
            st.session_state.messages.append({"role": "user", "content": f"Clarified to: *{selected_query}*"})
            
            # Generate RAG response
            with st.spinner("Querying knowledge base..."):
                response = st.session_state.pipeline.run(
                    query=selected_query,
                    faq_threshold=faq_threshold,
                    top_k=top_k,
                    temperature=temperature
                )
                
            # Store response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "retrieved_from": response["retrieved_from"],
                "sources": response["sources"]
            })
            st.rerun()
            
    if st.button("❌ Cancel clarification", key="cancel_clarify"):
        st.session_state.awaiting_clarification = False
        st.session_state.messages.append({"role": "assistant", "content": "Clarification cancelled. Please ask your question again."})
        st.rerun()

# -----------------------------------------------------------------------------
# Chat Input Handling
# -----------------------------------------------------------------------------
else:
    if prompt := st.chat_input("Ask a question about the project (e.g. onboarding, budget, database design)..."):
        # Append User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if the LM Studio is offline
        if not st.session_state.llm_client.health_check():
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "🔴 **Error**: LM Studio server is offline. Please make sure the local server is running on http://localhost:1234 and try again."
            })
            st.rerun()
            
        # Run Ambiguity Analysis
        with st.spinner("Analyzing intent..."):
            clarify_result = st.session_state.clarifier.check_ambiguity(prompt)
            
        if clarify_result["is_ambiguous"]:
            # Pause flow and trigger clarification options
            st.session_state.awaiting_clarification = True
            st.session_state.clarification_options = clarify_result["options"]
            st.session_state.original_query = prompt
            st.rerun()
        else:
            # Query is clear, run direct RAG Pipeline
            with st.spinner("Generating answer..."):
                response = st.session_state.pipeline.run(
                    query=prompt,
                    faq_threshold=faq_threshold,
                    top_k=top_k,
                    temperature=temperature
                )
                
            # Append Assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "retrieved_from": response["retrieved_from"],
                "sources": response["sources"]
            })
            st.rerun()
