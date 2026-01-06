import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os
import time
from datetime import datetime
from streamlit_option_menu import option_menu 
from PIL import Image, ImageFilter
import io
import sys
from pathlib import Path
import base64
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.unified_pipeline import UnifiedRAGPipeline
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Tech Chatbot Dashboard", page_icon="🤖", layout="wide")
load_dotenv()

THEME_OPTIONS = ["Light Mode", "Dark Mode"]
# Persist theme via URL to survive hard refreshes
try:
    _params = st.query_params
except Exception:
    _params = {}

_url_theme = None
if hasattr(_params, "get"):
    _theme_val = _params.get("theme")
    if isinstance(_theme_val, list):
        _url_theme = _theme_val[0] if _theme_val else None
    elif isinstance(_theme_val, str):
        _url_theme = _theme_val

if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = _url_theme if _url_theme in THEME_OPTIONS else THEME_OPTIONS[0]
if "bg_disabled" not in st.session_state:
    st.session_state["bg_disabled"] = False

def _persist_theme_to_url():
    try:
        st.query_params["theme"] = st.session_state.get("ui_theme", THEME_OPTIONS[0])
    except Exception:
        # Best-effort; ignore if running in environments without query params
        pass

ui_theme = st.session_state.get("ui_theme", THEME_OPTIONS[0])
if ui_theme not in THEME_OPTIONS:
    ui_theme = THEME_OPTIONS[0]
    st.session_state["ui_theme"] = ui_theme
_persist_theme_to_url()
is_dark = (ui_theme == "Dark Mode")

if is_dark:
    BG_COLOR = "#0E1117"
    CARD_BG = "rgba(38, 39, 48, 0.8)"
    TEXT_COLOR = "#FAFAFA"
    SUB_TEXT = "#A3A8B8"
    BORDER_COLOR = "rgba(255, 255, 255, 0.1)"
    ACCENT_COLOR = "#3B82F6"
    ICON_BOX_BG = "#1E3A8A"
    HEADER_BG = "linear-gradient(135deg, rgba(30, 58, 138, 0.9) 0%, rgba(23, 37, 84, 0.9) 100%)"
    PLOT_TEMPLATE = "plotly_dark"
    BG_OVERLAY = "rgba(14, 17, 23, 0.92)" 
else:
    BG_COLOR = "#E5E7EB"  # light gray base for better contrast
    CARD_BG = "rgba(255, 255, 255, 0.9)"
    TEXT_COLOR = "#1E293B"
    SUB_TEXT = "#374151"
    BORDER_COLOR = "rgba(0, 0, 0, 0.18)"
    ACCENT_COLOR = "#0A67C8"
    ICON_BOX_BG = "#1F3C72"
    HEADER_BG = "linear-gradient(135deg, rgba(10, 103, 200, 0.95) 0%, rgba(6, 64, 130, 0.95) 100%)"
    PLOT_TEMPLATE = "plotly_white"
    BG_OVERLAY = "rgba(229, 231, 235, 0.95)"
    HEADER_TEXT = "#0A2A5E"
    SUBHEADER_TEXT = "#1F2937"
    PLOT_COLOR_SEQ = ["#0A67C8", "#EF4444", "#10B981", "#6366F1"]

if is_dark:
    HEADER_TEXT = "#FFFFFF"
    SUBHEADER_TEXT = "rgba(255,255,255,0.85)"
    PLOT_COLOR_SEQ = ["#3B82F6", "#F59E0B", "#22D3EE", "#10B981"]

H1_GRADIENT_CSS = ""
if is_dark:
    H1_GRADIENT_CSS = f"""
        background: linear-gradient(120deg, {ACCENT_COLOR} 0%, #dbeafe 50%, {ACCENT_COLOR} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    """

# --- HÀM XỬ LÝ ẢNH NỀN ---
@st.cache_data
def get_blurred_background(image_path, blur_radius=10):
    try:
        img = Image.open(image_path)
        blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius)) # Làm mờ
        buffered = io.BytesIO()
        blurred_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        return None

# --- CẤU HÌNH BACKGROUND ---
base_dir = Path(__file__).resolve().parent
bg_image_path = base_dir / "background.png"  # 
bg_base64 = get_blurred_background(str(bg_image_path), blur_radius=3) if bg_image_path.exists() else None
# --- 2. CẤU HÌNH GIAO DIỆN & MÀU SẮC ---
bg_css_rule = ""
if bg_base64 and not st.session_state.get("bg_disabled", False):
    bg_css_rule = f"""
    .stApp {{
        background-image: linear-gradient({BG_OVERLAY}, {BG_OVERLAY}), url("data:image/png;base64,{bg_base64}");
        background-attachment: fixed;
        background-size: 100% auto;
        background-position: top center;
        background-repeat: no-repeat;
    }}
    """

# --- 3. CSS TÙY CHỈNH ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
    }}
    {bg_css_rule}
    .stTabs {{ display: none; }}
    .block-container {{ padding-top: 18px; }}
    /* Highlight icons in option menu */
    .nav-pills .nav-link i,
    .nav-pills .nav-link svg {{ color: {SUB_TEXT} !important; transition: color 0.2s ease; }}
    .nav-pills .nav-link.active i,
    .nav-pills .nav-link.active svg {{ color: {ACCENT_COLOR} !important; }}

    /* HEADER */
    .main-header {{
        background: {HEADER_BG};
        padding: 14px 18px;
        border-radius: 0;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid {BORDER_COLOR};
        backdrop-filter: blur(10px);
        position: relative;
    }}
    .main-header h1 {{
        color: {HEADER_TEXT} !important;
        font-weight: 900;
        font-size: 30px;
        line-height: 1.2;
        margin: 0 auto;
        max-width: 960px;
        width: 100%;
        text-align: center !important;
        letter-spacing: 0.2px;
        {H1_GRADIENT_CSS}
    }}
    .main-header p {{
        color: {SUBHEADER_TEXT} !important;
        font-size: 14px;
        margin: 6px auto 0 auto;
        max-width: 960px;
        width: 100%;
        text-align: center !important;
    }}

    /* KPI CARDS */
    .kpi-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 20px;
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 20px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
        backdrop-filter: blur(5px);
    }}
    .kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }}
    
    .icon-box {{
        width: 64px; height: 64px;
        background-color: {ICON_BOX_BG};
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 28px; color: white;
        flex-shrink: 0;
    }}
    .kpi-content {{ display: flex; flex-direction: column; }}
    .kpi-label {{ font-size: 14px; color: {SUB_TEXT}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ font-size: 26px; font-weight: 800; color: {TEXT_COLOR}; margin-top: 4px; }}

    /* CHAT UI */
    .chat-container {{ display: flex; flex-direction: column; gap: 16px; padding: 10px 0 150px 0; }} 
    .chat-row {{ display: flex; width: 100%; }}
    .row-user {{ justify-content: flex-end; }}
    .row-bot {{ justify-content: flex-start; }}
    .bubble {{ max-width: 80%; padding: 14px 18px; border-radius: 18px; font-size: 15px; line-height: 1.5; }}
    .bubble-user {{ background-color: {ACCENT_COLOR}; color: white; border-bottom-right-radius: 4px; }}
    .bubble-bot {{ background-color: {CARD_BG}; color: {TEXT_COLOR}; border: 1px solid {BORDER_COLOR}; border-bottom-left-radius: 4px; }}
    .bot-avatar {{ width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 24px; }}

    /* SUGGESTION BUTTONS */
    .suggestion-btn button {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 20px;
        padding: 10px 20px;
        color: {TEXT_COLOR};
        font-weight: 500;
        backdrop-filter: blur(5px);
    }}
    .suggestion-btn button:hover {{
        border-color: {ACCENT_COLOR};
        color: {ACCENT_COLOR};
    }}

    /* 1. INPUTS & BUTTONS CƠ BẢN */
    .stTextInput input {{ background-color: {CARD_BG}; color: {TEXT_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 10px; padding: 12px; }}
    div.stButton > button {{ background-color: {ACCENT_COLOR}; color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.5rem 1rem; }}
    [data-testid="stDataFrame"] {{ background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; overflow: hidden; }}
    div[data-testid="stForm"] {{ background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}

    /* 2. ẨN KHUNG DƯỚI CÙNG */
    [data-testid="stBottom"] {{
        background: transparent !important;
        padding: 0 !important; 
        box-shadow: none !important;
    }}

    /* 3. ĐỊNH VỊ KHUNG CHAT (FIX LỖI ĐEN NỀN TẠI ĐÂY) */
    [data-testid="stChatInput"] {{
        background: transparent !important; /* Xóa màu nền cha */
        background-color: transparent !important;
        position: fixed !important;
        bottom: 40px !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1000 !important;
        pointer-events: none;
    }}

    [data-testid="stChatInput"] > div {{
        max-width: 1100px !important;
        width: 90% !important;
        margin: 0 auto !important;
        pointer-events: auto;
    }}

    /* 4. TẠO HÌNH VIÊN THUỐC KÍNH MỜ */
    [data-testid="stChatInput"] form {{
        position: relative !important;
        background: rgba(20, 20, 20, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        padding: 5px 10px !important;
        overflow: hidden !important;
    }}

    /* Hiệu ứng nền phát sáng bên trong */
    [data-testid="stChatInput"] form:before {{
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        border-radius: 50px;
        background: radial-gradient(circle at 30% 30%, {ACCENT_COLOR}33, transparent 45%),
                    radial-gradient(circle at 80% 70%, #22d3ee26, transparent 50%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    /* Khi click vào thì sáng lên */
    [data-testid="stChatInput"] form:focus-within {{
        box-shadow: 0 22px 48px rgba(0,0,0,0.3) !important;
        border-color: {ACCENT_COLOR}66 !important;
    }}
    [data-testid="stChatInput"] form:focus-within::before {{
        opacity: 1;
    }}

    /* 5. TINH CHỈNH TEXTAREA & BUTTON */
    [data-testid="stChatInput"] textarea {{
        color: white !important;
        font-size: 16px !important;
    }}
    [data-testid="stChatInput"] button {{
        border-radius: 50% !important;
        background: {ACCENT_COLOR} !important;
    }}

    /* 6. NÚT XÓA MÀU ĐỎ */
    div.stButton > button:has(div p:contains('🗑️')) {{
        background-color: #EF4444 !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2);
    }}
    div.stButton > button:has(div p:contains('🗑️')):hover {{
        background-color: #DC2626 !important;
        box-shadow: 0 6px 12px rgba(239, 68, 68, 0.3);
    }}
</style>
""", unsafe_allow_html=True)

# --- HEADER HTML ---
st.markdown("""
    <div class="main-header">
        <h1>Chatbot công nghệ & Phân tích dữ liệu</h1>
        <p>Trợ lý ảo hỗ trợ tra cứu sản phẩm & Phân tích xu hướng thị trường 2026</p>
    </div>
""", unsafe_allow_html=True)

# --- 4. HÀM HỖ TRỢ ---
def style_fig(fig, show_grid=None):
    if show_grid is None:
        show_grid = not is_dark
    grid_color = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.16)"
    text_color = "#FAFAFA" if is_dark else "#1E293B"
    tick_color = "#A3A8B8" if is_dark else "#64748B"
    fig.update_layout(template=PLOT_TEMPLATE, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=50, b=20), font=dict(family="Inter", size=12, color=tick_color), title_font=dict(size=18, color=text_color, family="Inter", weight=700))
    fig.update_xaxes(showgrid=show_grid, gridcolor=grid_color, zeroline=False, showline=True, linecolor=grid_color, tickfont=dict(color=tick_color))
    fig.update_yaxes(showgrid=show_grid, gridcolor=grid_color, zeroline=False, showline=False, tickfont=dict(color=tick_color))
    return fig

def extract_brands(title):
    brands = {'Apple': ['iPhone', 'iPad', 'MacBook', 'Apple'], 'Samsung': ['Samsung', 'Galaxy'], 'Xiaomi': ['Xiaomi', 'Redmi', 'POCO'], 'Oppo': ['OPPO', 'Reno'], 'Vivo': ['vivo'], 'Sony': ['Sony'], 'Dell': ['Dell'], 'Asus': ['Asus'], 'HP': ['HP'], 'Logitech': ['Logitech']}
    for b, k in brands.items():
        for key in k:
            if key.lower() in str(title).lower(): return b
    return 'Other'

@st.cache_data
def load_data():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_DB_HOST"),
        port=os.getenv("POSTGRES_DB_PORT"),
        dbname=os.getenv("POSTGRES_DB_NAME"),
        user=os.getenv("POSTGRES_DB_USER"),
        password=os.getenv("POSTGRES_DB_PASSWORD"),
    )
    df = pd.read_sql(
        "SELECT full_title AS title, category, price, content_type, crawl_date, "
        "source_url AS url, content_text AS text "
        "FROM products ORDER BY crawl_date DESC LIMIT 3000;",
        conn,
    )
    conn.close()
    return df

# --- 5. INIT DATA ---
if 'df' not in st.session_state:
    try:
        st.session_state.df = load_data()
        st.session_state.data_error = None
    except Exception as e:
        st.session_state.data_error = str(e)
        st.session_state.df = pd.DataFrame(
            columns=["title", "category", "price", "content_type", "crawl_date", "url", "text"]
        )
    if st.session_state.df.empty:
        st.session_state.df['crawl_date'] = pd.to_datetime([])
        st.session_state.df['brand'] = pd.Series(dtype="object")
        st.session_state.df['price_numeric'] = pd.Series(dtype="float")
        st.session_state.df['text_length'] = pd.Series(dtype="int")
    else:
        st.session_state.df['crawl_date'] = pd.to_datetime(st.session_state.df['crawl_date'])
        st.session_state.df['brand'] = st.session_state.df['title'].apply(extract_brands)
        st.session_state.df['price_numeric'] = pd.to_numeric(st.session_state.df['price'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')
        st.session_state.df['text_length'] = st.session_state.df['text'].astype(str).str.len()

df = st.session_state.df.copy()

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    if st.session_state.get("data_error"):
        st.warning("Khong the ket noi Postgres. Vui long kiem tra .env va database.")
    st.caption("Bộ lọc dữ liệu đã được chuyển lên đầu tab Phân tích.")
    st.caption("Theme controls are now in the Settings tab.")
    st.divider()
    st.markdown("© 2026 Tech Chatbot Pro v5.0")

# --- 7. NAVIGATION ---
menu_options = ["Chatbot", "Phân tích", "Dữ liệu", "Settings"]

if "nav_index" not in st.session_state:
    st.session_state.nav_index = 0
elif st.session_state.nav_index >= len(menu_options):
    st.session_state.nav_index = 0

# --- FIX LỖI Ở ĐÂY: Thêm tham số `key` ---
def on_menu_change(key):
    selection = st.session_state[key]
    st.session_state.nav_index = menu_options.index(selection)

selected = option_menu(
    menu_title=None,
    options=menu_options,
    icons=["robot", "bar-chart-fill", "table", "gear"],
    default_index=st.session_state.nav_index,
    orientation="horizontal",
    key="main_navigation",
    on_change=on_menu_change, # Callback đã sửa
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "transparent",
            "margin-bottom": "12px",
            "position": "static",
        },
        "icon": {"color": SUB_TEXT, "font-size": "20px"},
        "nav-link": {
            "font-family": "Inter",
            "font-size": "16px",
            "text-align": "center",
            "margin": "0 6px",
            "color": SUB_TEXT,
            "font-weight": "700",
            "border-radius": "12px",
            "padding": "10px 14px",
        },
        "nav-link-selected": {
            "background-color": "transparent",
            "color": TEXT_COLOR,
            "border": f"2px solid {ACCENT_COLOR}",
            "box-shadow": "0 2px 6px rgba(0,0,0,0.08)"
        }
    }
)

# ============ VIEW 1: CHATBOT ============
if selected == "Chatbot":
    col_chat_title, col_chat_reset = st.columns([8, 1])
    
    with col_chat_title:
        st.markdown(f"<h3 style='color:{TEXT_COLOR}; margin:0;'>🤖 Trợ lý AI & Sản phẩm</h3>", unsafe_allow_html=True)
        st.caption("Tra cứu thông tin, so sánh giá và phân tích xu hướng.")
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = UnifiedRAGPipeline()
        st.session_state.rag_pipeline.llm_service.generate_search_queries = lambda q: [q]
    if "rag_top_k" not in st.session_state:
        st.session_state.rag_top_k = TOP_K_RETRIEVE
    if "rag_top_n" not in st.session_state:
        st.session_state.rag_top_n = TOP_N_RERANK
        
    with col_chat_reset:
        if st.button("🗑️", use_container_width=True, help="Xóa lịch sử trò chuyện"):
            st.session_state.chat_history = []
            if "rag_pipeline" in st.session_state:
                st.session_state.rag_pipeline.clear_history()
            st.rerun()
    
    GEMINI_MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL", "3"))
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    GEMINI_BACKOFF_BASE = float(os.getenv("GEMINI_BACKOFF_BASE", "3"))

    def _ask_gemini_legacy(q, ctx):
        now = time.time()
        last_ts = st.session_state.get("last_gemini_ts", 0.0)
        wait_sec = GEMINI_MIN_INTERVAL - (now - last_ts)
        if wait_sec > 0:
            return f"Vui long doi {wait_sec:.1f}s truoc khi gui cau hoi tiep."
        try:
            api = os.getenv("GEMINI_API_KEY")
            if not api: return "Vui lòng cấu hình GEMINI_API_KEY."
            genai.configure(api_key=api)
            st.session_state["last_gemini_ts"] = now
            model = genai.GenerativeModel("gemini-2.0-flash")
            info = "\n".join([f"- {r['full_title']} ({r['price']})" for _, r in ctx.iterrows()]) if not ctx.empty else "Không tìm thấy dữ liệu DB cụ thể."
            for attempt in range(GEMINI_MAX_RETRIES + 1):
                try:
                    return model.generate_content(f"Context: {info}\nUser question: {q}\nAnswer in Vietnamese helpful manner:").text
                except Exception as e:
                    err = str(e).lower()
                    if attempt < GEMINI_MAX_RETRIES and ("429" in err or "quota" in err or "rate" in err):
                        time.sleep(GEMINI_BACKOFF_BASE * (2 ** attempt))
                        continue
                    raise
        except Exception as e: return f"Lỗi: {e}"

    def ask_gemini(q, ctx):
        now = time.time()
        last_ts = st.session_state.get("last_gemini_ts", 0.0)
        wait_sec = GEMINI_MIN_INTERVAL - (now - last_ts)
        if wait_sec > 0:
            return {
                "answer": f"Vui long doi {wait_sec:.1f}s truoc khi gui cau hoi tiep.",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0,
            }

        try:
            pipeline = st.session_state.get("rag_pipeline")
            if pipeline is None:
                pipeline = UnifiedRAGPipeline()
                st.session_state["rag_pipeline"] = pipeline

            st.session_state["last_gemini_ts"] = now
            top_k = st.session_state.get("rag_top_k", TOP_K_RETRIEVE)
            top_n = st.session_state.get("rag_top_n", TOP_N_RERANK)

            for attempt in range(GEMINI_MAX_RETRIES + 1):
                try:
                    return pipeline.query(query=q, top_k=top_k, top_n=top_n, use_history=True)
                except Exception as e:
                    err = str(e).lower()
                    if attempt < GEMINI_MAX_RETRIES and ("429" in err or "quota" in err or "rate" in err):
                        time.sleep(GEMINI_BACKOFF_BASE * (2 ** attempt))
                        continue
                    raise
            return {
                "answer": "RAG pipeline failed to respond.",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0,
            }
        except Exception as e:
            return {
                "answer": f"Loi RAG: {e}",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0,
            }

    chat_container = st.container()
    
    with chat_container:
        if len(st.session_state.chat_history) == 0:
            st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: {TEXT_COLOR}; font-weight: 700;'>Xin chào! Tôi có thể giúp gì cho bạn?</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: {SUB_TEXT};'>Hãy thử chọn một trong các gợi ý bên dưới:</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            c_pad_l, c1, c2, c3, c_pad_r = st.columns([1, 2, 2, 2, 1])
            st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
            with c1:
                if st.button("Top 5 điện thoại giá rẻ", use_container_width=True):
                    st.session_state.pending_prompt = "Top 5 điện thoại giá rẻ"
                    st.rerun()
            with c2:
                if st.button("Laptop cho sinh viên", use_container_width=True):
                    st.session_state.pending_prompt = "Laptop cho sinh viên"
                    st.rerun()
            with c3:
                if st.button("Xu hướng giá iPhone", use_container_width=True):
                    st.session_state.pending_prompt = "Xu hướng giá iPhone"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            for msg in st.session_state.chat_history:
                role = "row-user" if msg["role"] == "user" else "row-bot"
                bubble = "bubble-user" if msg["role"] == "user" else "bubble-bot"
                avatar = "" if msg["role"] == "user" else '<div class="bot-avatar">🤖</div>'
                st.markdown(f'<div class="chat-row {role}">{avatar}<div class="bubble {bubble}">{msg["content"]}</div></div>', unsafe_allow_html=True)
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("Nguon trich dan"):
                        for i, source in enumerate(msg["sources"], 1):
                            title = source.get("title", "Unknown")
                            url = source.get("url", "")
                            st.markdown(f"**{i}. {title}**")
                            if url:
                                st.markdown(f"[{url}]({url})")
                if msg["role"] == "assistant" and msg.get("retrieved_count") is not None:
                    retrieved = msg.get("retrieved_count", 0)
                    reranked = msg.get("reranked_count", 0)
                    st.caption(f"Retrieved: {retrieved} | Reranked: {reranked}")
            
            st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

    pending_prompt = st.session_state.get("pending_prompt")
    chat_prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")
    prompt = pending_prompt or chat_prompt
    if prompt:
        st.session_state.pending_prompt = None
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        result = ask_gemini(prompt, None)
        answer = result.get("answer", "Xin lỗi, không thể trả lời lúc này.")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources", []),
            "retrieved_count": result.get("retrieved_count", 0),
            "reranked_count": result.get("reranked_count", 0),
        })
        st.rerun()

# ============ VIEW 2: PHÂN TÍCH ============
elif selected == "Phân tích":
    if df.empty:
        st.info("Chưa có dữ liệu để phân tích.")
    else:
        if "analysis_filter_dates" not in st.session_state:
            st.session_state["analysis_filter_dates"] = [
                df['crawl_date'].min().date(),
                df['crawl_date'].max().date(),
            ]
        if "analysis_filter_cats" not in st.session_state:
            cat_list = sorted(df['category'].dropna().unique().tolist())
            st.session_state["analysis_filter_cats"] = cat_list[:3] if len(cat_list) > 3 else cat_list

        st.markdown("### Bộ lọc dữ liệu")
        available_cats = sorted(df['category'].dropna().unique().tolist())
        with st.form("analysis_filter"):
            col1, col2 = st.columns([2, 3])
            with col1:
                dates = st.date_input("Thời gian", value=st.session_state["analysis_filter_dates"])
            with col2:
                cats_selected = st.multiselect(
                    "Danh mục",
                    available_cats,
                    default=st.session_state["analysis_filter_cats"],
                )
            submitted = st.form_submit_button("Áp dụng bộ lọc")

        if submitted:
            st.session_state["analysis_filter_dates"] = dates
            st.session_state["analysis_filter_cats"] = cats_selected

        dates = st.session_state["analysis_filter_dates"]
        cats_selected = st.session_state["analysis_filter_cats"]
        if not cats_selected:
            cats_selected = available_cats

        analysis_df = df[
            (df['crawl_date'].dt.date >= dates[0])
            & (df['crawl_date'].dt.date <= dates[1])
            & (df['category'].isin(cats_selected))
        ]

        st.caption(f"Đang lọc {len(analysis_df)} / {len(df)} bản ghi.")

        if analysis_df.empty:
            st.info("Không có dữ liệu thỏa bộ lọc.")
        else:
            st.markdown("### Tổng quan")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""<div class="kpi-card"><div class="icon-box">📄</div><div class="kpi-content"><div class="kpi-label">Tổng bản ghi</div><div class="kpi-value">{len(analysis_df):,.0f}</div></div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="kpi-card"><div class="icon-box">🏷️</div><div class="kpi-content"><div class="kpi-label">Tổng sản phẩm</div><div class="kpi-value">{len(analysis_df[analysis_df["content_type"]=="product"]):,.0f}</div></div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="kpi-card"><div class="icon-box">💲</div><div class="kpi-content"><div class="kpi-label">Giá trung bình</div><div class="kpi-value">₫{analysis_df["price_numeric"].mean():,.0f}</div></div></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                vc = analysis_df['category'].value_counts().head(8).reset_index()
                vc.columns = ['category', 'count']
                st.plotly_chart(
                    style_fig(
                        px.bar(
                            vc,
                            y='category',
                            x='count',
                            orientation='h',
                            text='count',
                            color_discrete_sequence=PLOT_COLOR_SEQ,
                            title="Top Danh mục phổ biến",
                        )
                    ),
                    use_container_width=True,
                )
            with c2:
                vb = analysis_df['brand'].value_counts().head(8).reset_index()
                vb.columns = ['brand', 'count']
                st.plotly_chart(
                    style_fig(
                        px.bar(
                            vb,
                            y='brand',
                            x='count',
                            orientation='h',
                            text='count',
                            color_discrete_sequence=PLOT_COLOR_SEQ,
                            title="Top Thương hiệu",
                        )
                    ),
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown("### Phân tích sâu")
            prod_df = analysis_df[analysis_df['content_type'] == 'product']

            if not prod_df.empty:
                st.markdown("#### 1. Cấu trúc thị trường (Sunburst)")
                fig_sun = px.sunburst(
                    prod_df,
                    path=['category', 'brand'],
                    values='price_numeric',
                    color='price_numeric',
                    color_continuous_scale='Blues',
                )
                st.plotly_chart(style_fig(fig_sun), use_container_width=True)
                st.markdown("---")

                st.markdown("#### 2. Phân bố khoảng giá (Box Plot)")
                fig_box = px.box(prod_df, x='category', y='price_numeric', color='category', color_discrete_sequence=PLOT_COLOR_SEQ)
                st.plotly_chart(style_fig(fig_box, show_grid=True), use_container_width=True)
                st.markdown("---")

                st.markdown("#### 3. Tần suất phân bố giá (Histogram)")
                fig_hist = px.histogram(prod_df, x="price_numeric", nbins=30, color="category", marginal="box", color_discrete_sequence=PLOT_COLOR_SEQ)
                st.plotly_chart(style_fig(fig_hist, show_grid=True), use_container_width=True)
                st.markdown("---")

                st.markdown("#### 4. Top 10 Sản phẩm giá cao nhất")
                top_10 = prod_df.nlargest(10, 'price_numeric').sort_values('price_numeric', ascending=True)
                fig_top = px.bar(top_10, x='price_numeric', y='title', orientation='h', text='price_numeric', color='category', color_discrete_sequence=PLOT_COLOR_SEQ)
                st.plotly_chart(style_fig(fig_top, show_grid=True), use_container_width=True)
                st.markdown("---")

                st.markdown("#### 5. Bản đồ nhiệt: Thương hiệu x Danh mục")
                heat = prod_df.groupby(['category', 'brand']).size().reset_index(name='count')
                fig_heat = px.density_heatmap(heat, x='brand', y='category', z='count', color_continuous_scale='Magma', text_auto=True)
                st.plotly_chart(style_fig(fig_heat), use_container_width=True)
            else:
                st.info("Chưa có dữ liệu sản phẩm để phân tích.")

            st.markdown("---")
            st.markdown("### Dữ liệu thu thập theo thời gian")
            daily = analysis_df.groupby(analysis_df['crawl_date'].dt.date).size().reset_index(name='count')
            fig = px.line(daily, x='crawl_date', y='count', markers=True, line_shape='spline', color_discrete_sequence=PLOT_COLOR_SEQ)
            fig.update_traces(fill='tozeroy')
            st.plotly_chart(style_fig(fig, show_grid=True), use_container_width=True)

# ============ VIEW 3: DỮ LIỆU ============
elif selected == "Dữ liệu":
    st.subheader("Bảng dữ liệu chi tiết")
    s = st.text_input("Tìm kiếm nhanh:", placeholder="Nhập tên sản phẩm...", key="search_data")
    df_show = df[df['title'].str.contains(s, case=False, na=False)] if s else df
    st.caption(f"Đang hiển thị **{len(df_show)}** bản ghi.")
    st.dataframe(df_show[['title','brand','category','price','crawl_date','url']], use_container_width=True, hide_index=True, height=700)

# ============ VIEW 4: SETTINGS ============
elif selected == "Settings":
    st.subheader("Settings")
    st.caption("Thay đổi theme và tinh chỉnh chatbot tại đây.")

    st.radio(
        "Theme",
        THEME_OPTIONS,
        index=THEME_OPTIONS.index(st.session_state.get("ui_theme", THEME_OPTIONS[0])),
        horizontal=True,
        key="ui_theme",
        on_change=_persist_theme_to_url,
    )
    st.checkbox(
        "Tắt nền trang",
        value=st.session_state.get("bg_disabled", False),
        key="bg_disabled",
    )

    st.divider()
    st.markdown("**Chatbot retrieval**")
    col1, col2 = st.columns(2)
    with col1:
        top_k_default = int(st.session_state.get("rag_top_k", TOP_K_RETRIEVE))
        st.slider("Top K retrieve", 1, 50, value=top_k_default, key="rag_top_k")
    with col2:
        max_top_n = max(1, int(st.session_state.get("rag_top_k", TOP_K_RETRIEVE)))
        top_n_default = int(st.session_state.get("rag_top_n", TOP_N_RERANK))
        st.slider(
            "Top N rerank",
            1,
            max_top_n,
            value=min(top_n_default, max_top_n),
            key="rag_top_n",
        )
    st.caption("Các thay đổi áp dụng cho toàn bộ ứng dụng.")
