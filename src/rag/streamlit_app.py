"""
Streamlit UI for RAG Chatbot
Uses SmartRAGPipeline with Fast and Deep modes
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
from rag.smart_pipeline import SmartRAGPipeline
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK


def initialize_session_state():
    """Initialize session state variables"""
    if "pipeline" not in st.session_state:
        # Use SmartRAGPipeline with Fast/Deep mode support
        st.session_state.pipeline = SmartRAGPipeline()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_RETRIEVE
    
    if "top_n" not in st.session_state:
        st.session_state.top_n = TOP_N_RERANK
    
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "fast"


def clean_json_string(json_str: str) -> str:
    """
    Clean potential issues in JSON string from LLM.
    - Fix trailing commas
    - Remove markdown fences if present
    """
    # Remove markdown fences just in case
    json_str = json_str.replace("```json", "").replace("```", "")
    
    # Remove trailing commas before } or ]
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
                chart_spec = json.loads(cleaned_json, strict=False)
                
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
                        chart_spec = json.loads(cleaned_json, strict=False)
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
    # Filter sources that have images and are strictly PRODUCTS (not articles)
    # This improves relevance significantly (e.g. MacBook M1 vs M4)
    product_sources = [
        s for s in sources 
        if s.get("images") and isinstance(s.get("images"), list) and len(s["images"]) > 0
        and s.get("content_type") == "product"
    ]
    
    # Heuristic to find the best image (not a banner/logo)
    def get_best_image(img_list):
        bad_keywords = ["banner", "logo", "icon", "thumb", "fallback", "promotion", "quang-cao"]
        # Look for images that look like actual product shots (e.g. contain 'Products/Images' for TGDD)
        for url in img_list:
            u_low = url.lower()
            if "products/images" in u_low and not any(k in u_low for k in bad_keywords):
                return url
        
        # Fallback to first non-bad keyword if possible
        for url in img_list:
            if not any(k in url.lower() for k in bad_keywords):
                return url
                
        return img_list[0] # ultimate fallback
    
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
    display_count = min(len(unique_products), 4)
    cols = st.columns(display_count)
    
    for i in range(display_count):
        prod = unique_products[i]
        with cols[i]:
            # Get best image
            img_url = get_best_image(prod["images"])
            st.image(img_url, width="stretch")
            
            # Title & Price
            title = prod['title']
            # Fix common Vietnamese typo/missing accents in title
            if title.lower().startswith("san pham"):
                title = "Sản phẩm" + title[8:]
            elif title.lower().startswith("sản pham"):
                 title = "Sản phẩm" + title[8:]
                 
            st.markdown(f"**{title}**")
            if prod.get("price"):
                price = prod['price']
                if isinstance(price, (int, float)):
                    st.caption(f"💰 {price:,} VNĐ")
                else:
                    # Check if string is digit-only
                    if str(price).replace('.', '').isdigit():
                        try:
                            st.caption(f"💰 {int(float(price)):,} VNĐ")
                        except:
                            st.caption(f"💰 {price} VNĐ")
                    else:
                        st.caption(f"💰 {price} VNĐ")
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
    
    st.title("Tech Chatbot - Hệ thống RAG")
    st.markdown("Hỏi về sản phẩm công nghệ và bài viết đánh giá!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Cấu hình")
        
        # Search mode toggle
        st.markdown("### Chế độ tìm kiếm")
        mode_option = st.radio(
            "Chọn chế độ:",
            ["⚡ Fast (Nhanh)", "🔍 Deep (Agentic)"],
            index=0 if st.session_state.search_mode == "fast" else 1,
            help="Fast: Tìm kiếm đơn giản, nhanh. Deep: Suy luận nhiều bước, phù hợp câu hỏi phức tạp."
        )
        st.session_state.search_mode = "fast" if "Fast" in mode_option else "deep"
        
        if st.session_state.search_mode == "deep":
            st.info("🤖 Chế độ Deep sử dụng AI agent để suy luận nhiều bước. Có thể mất thêm thời gian.")
        
        st.markdown("---")
        
        top_k = st.slider(
            "Top K (Tìm kiếm)",
            min_value=5,
            max_value=50,
            value=st.session_state.top_k,
            help="Số lượng tài liệu truy xuất từ database"
        )
        st.session_state.top_k = top_k
        
        top_n = st.slider(
            "Top N (Xếp hạng)",
            min_value=3,
            max_value=20,
            value=st.session_state.top_n,
            help="Số lượng tài liệu sau khi xếp hạng lại"
        )
        st.session_state.top_n = top_n
        
        if st.button("Xóa lịch sử trò chuyện"):
            st.session_state.pipeline.clear_history()
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Thông tin hệ thống")
        mode_display = "Deep (Agentic)" if st.session_state.search_mode == "deep" else "Fast"
        st.info(f"Chế độ: {mode_display}\nTruy xuất: Top {top_k} → Xếp hạng: Top {top_n}")
    
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
                    with st.expander("Nguồn trích dẫn"):
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
    if prompt := st.chat_input("Hỏi về sản phẩm công nghệ..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            spinner_text = "Đang suy nghĩ..." if st.session_state.search_mode == "fast" else "Đang suy luận (Deep mode)..."
            with st.spinner(spinner_text):
                try:
                    result = st.session_state.pipeline.query(
                        query=prompt,
                        mode=st.session_state.search_mode,
                        top_k=st.session_state.top_k,
                        top_n=st.session_state.top_n,
                        use_history=True
                    )
                    
                    answer = result.get("answer", "Xin lỗi, tôi không thể tạo phản hồi.")
                    sources = result.get("sources", [])
                    mode_used = result.get("mode", "fast")
                    
                    # 1. Render Gallery
                    render_product_gallery(sources)
                    
                    # 2. Render Response (Charts + Text)
                    render_response_with_charts(answer)
                    
                    # 3. Sources Expander
                    if sources:
                        with st.expander(f"Nguồn trích dẫn ({len(sources)} tài liệu)"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source.get('title', 'Unknown')}**")
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
                    mode_label = "🔍 Deep" if mode_used == "deep" else "⚡ Fast"
                    if mode_used == "deep":
                        st.caption(f"{mode_label} | Iterations: {result.get('iterations', 0)}")
                    else:
                        st.caption(f"{mode_label} | Tìm kiếm: {result.get('retrieved_count', 0)} | "
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