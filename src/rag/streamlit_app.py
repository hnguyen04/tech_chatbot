"""
Streamlit UI for RAG Chatbot
Uses UnifiedRAGPipeline with existing Vietnamese embeddings
"""
import sys
import os
import json
import re
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
        st.session_state.pipeline = UnifiedRAGPipeline()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_RETRIEVE
    
    if "top_n" not in st.session_state:
        st.session_state.top_n = TOP_N_RERANK


def clean_json_string(json_str: str) -> str:
    """
    Clean potential issues in JSON string from LLM.
    - Remove // comments
    - Fix trailing commas
    """
    # Remove // comments
    json_str = re.sub(r"//.*", "", json_str)
    
    # Remove trailing commas before } or ]
    # This is a basic regex, might not cover all edge cases but helps with common LLM errors
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    
    return json_str.strip()


import traceback

def render_response_with_charts(text: str):
    """
    Render text that might contain Vega-Lite JSON blocks.
    """
    # Regex to find json:vega-lite blocks
    # Matches ```json:vega-lite ... ``` (non-greedy)
    pattern = r"```json:vega-lite\s+(.*?)\s+```"
    
    parts = re.split(pattern, text, flags=re.DOTALL)
    
    # Parts will be [text, json_str, text, json_str, ...]
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text part
            if part.strip():
                st.markdown(part)
        else:
            # JSON part
            cleaned_json = ""
            try:
                # Clean up JSON
                cleaned_json = clean_json_string(part)
                chart_spec = json.loads(cleaned_json)
                
                # Basic validation
                if not isinstance(chart_spec, dict):
                    raise ValueError(f"Chart specification must be a dictionary, got {type(chart_spec)}")
                
                # Render
                st.vega_lite_chart(chart_spec, width="stretch")
            except json.JSONDecodeError as e:
                # Fallback: Try to find the first { and last }
                try:
                    start = part.find("{")
                    end = part.rfind("}")
                    if start != -1 and end != -1:
                        sub_part = part[start:end+1]
                        cleaned_json = clean_json_string(sub_part)
                        chart_spec = json.loads(cleaned_json)
                        if not isinstance(chart_spec, dict):
                             raise ValueError(f"Chart specification must be a dictionary, got {type(chart_spec)}")
                        st.vega_lite_chart(chart_spec, width="stretch")
                    else:
                        raise e
                except Exception as inner_e:
                    st.error(f"Failed to render chart: {type(e).__name__}: {str(e)}")
                    with st.expander("Debug Chart Error"):
                        st.text("Original Part:")
                        st.code(part, language="json")
                        st.text("Cleaned JSON:")
                        st.code(cleaned_json, language="json")
                        st.text("Traceback:")
                        st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Failed to render chart: {type(e).__name__}: {str(e)}")
                with st.expander("Debug Chart Error"):
                    st.text("Cleaned JSON:")
                    st.code(cleaned_json, language="json")
                    if 'chart_spec' in locals():
                        st.text("Parsed Object:")
                        st.write(chart_spec)
                    st.text("Traceback:")
                    st.code(traceback.format_exc())


def render_product_gallery(sources: list):
    """
    Render a horizontal gallery of products with images.
    """
    # Filter sources that have images and are products
    product_sources = [
        s for s in sources 
        if s.get("images") and isinstance(s.get("images"), list) and len(s["images"]) > 0
    ]
    
    # Deduplicate by record_id
    seen_ids = set()
    unique_products = []
    for p in product_sources:
        rid = p.get("record_id")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            unique_products.append(p)
    
    if not unique_products:
        return

    st.markdown("### 🛍️ Sản phẩm liên quan")
    
    # Display up to 4 products
    cols = st.columns(min(len(unique_products), 4))
    
    for i, col in enumerate(cols):
        prod = unique_products[i]
        with col:
            # Get first image
            img_url = prod["images"][0]
            st.image(img_url, width="stretch")
            
            # Title & Price
            st.markdown(f"**{prod['title']}**")
            if prod.get("price"):
                st.caption(f"💰 {prod['price']:,} VNĐ")
            else:
                st.caption("Liên hệ")
            
            # Link
            if prod.get("url"):
                st.markdown(f"[Xem chi tiết]({prod['url']})")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Tech Chatbot RAG",
        page_icon="🤖",
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # For assistant, we might have rich content
            if message["role"] == "assistant":
                # 1. Render Gallery (if sources exist in message)
                if "sources" in message:
                    render_product_gallery(message["sources"])
                
                # 2. Render Text/Charts
                render_response_with_charts(message["content"])
                
                # 3. Sources Expander
                if "sources" in message:
                    with st.expander("Sources Details"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**{i}. {source['title']}**")
                            if source.get("url"):
                                st.markdown(f"[{source['url']}]({source['url']})")
                            st.markdown(f"{source.get('content_preview', '')[:200]}...")
                            st.markdown("---")
            else:
                # User message is just text
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about tech products..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
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
                    
                    # 1. Render Gallery
                    render_product_gallery(sources)
                    
                    # 2. Render Response (Charts + Text)
                    render_response_with_charts(answer)
                    
                    # 3. Sources Expander
                    if sources:
                        with st.expander(f"Sources ({len(sources)} documents)"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                if source.get("url"):
                                    st.markdown(f"[{source['url']}]({source['url']})")
                                st.markdown(f"{source.get('content_preview', '')[:200]}...")
                                st.markdown("---")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Metadata
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