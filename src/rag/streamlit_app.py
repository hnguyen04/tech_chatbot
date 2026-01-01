"""
Streamlit UI for RAG Chatbot
Uses UnifiedRAGPipeline with existing Vietnamese embeddings
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
from rag.unified_pipeline import UnifiedRAGPipeline
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK


def initialize_session_state():
    """Initialize session state variables"""
    if "pipeline" not in st.session_state:
        # Use UnifiedRAGPipeline with existing Vietnamese embeddings
        # Default collection: "tech_embeddings" (can be changed via .env)
        st.session_state.pipeline = UnifiedRAGPipeline()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_RETRIEVE
    
    if "top_n" not in st.session_state:
        st.session_state.top_n = TOP_N_RERANK


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Tech Chatbot RAG",
        page_icon=None,
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("Tech Chatbot - RAG System")
    st.markdown("Ask questions about technology products and articles!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        top_k = st.slider(
            "Top K (Retrieval)",
            min_value=5,
            max_value=50,
            value=st.session_state.top_k,
            help="Number of documents to retrieve"
        )
        st.session_state.top_k = top_k
        
        top_n = st.slider(
            "Top N (Reranking)",
            min_value=3,
            max_value=20,
            value=st.session_state.top_n,
            help="Number of documents after reranking"
        )
        st.session_state.top_n = top_n
        
        if st.button("Clear Chat History"):
            st.session_state.pipeline.clear_history()
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### System Info")
        st.info(f"Retrieval: Top {top_k} → Reranking: Top {top_n}")
    
    # Main chat interface
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source['title']}**")
                        if source.get("url"):
                            st.markdown(f"[{source['url']}]({source['url']})")
                        st.markdown(f"{source.get('content_preview', '')[:200]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about tech products..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.pipeline.query(
                        query=prompt,
                        top_k=st.session_state.top_k,
                        top_n=st.session_state.top_n,
                        use_history=True
                    )
                    
                    answer = result.get("answer", "Sorry, I couldn't generate a response.")
                    sources = result.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"Sources ({len(sources)} documents)"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                if source.get("url"):
                                    st.markdown(f"[{source['url']}]({source['url']})")
                                st.markdown(f"{source.get('content_preview', '')[:200]}...")
                                st.markdown("---")
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Show metadata
                    st.caption(f"Retrieved: {result.get('retrieved_count', 0)} | "
                              f"Reranked: {result.get('reranked_count', 0)}")
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()