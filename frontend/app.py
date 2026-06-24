"""
app.py
------
Streamlit frontend for the Spheretech AI Assistant.
Connects the full RAG pipeline:
    User query → Retriever → LLM (Groq) → Response

Run with:
    streamlit run frontend/app.py
"""

import sys
import os
import json
import streamlit as st
import markdown as md_parser
from dotenv import load_dotenv

# Load .env file — must be before any API calls
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))

from retriever import Retriever
from llm_client import generate_answer, generate_answer_stream

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spheretech AI Assistant",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background & text ── */
    .stApp, .stApp > div, section[data-testid="stSidebar"] {
        background-color: #0B0F1A !important;
        color: #E2E8F0 !important;
    }

    /* ── Main content area ── */
    .block-container {
        background-color: #0B0F1A !important;
        padding-top: 2rem !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }

    /* ── All text elements ── */
    p, h1, h2, h3, h4, span, label, div {
        color: #E2E8F0 !important;
    }

    /* ── Header badge ── */
    .sphere-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1E3A5F, #0EA5E9);
        color: white !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
    }
    .sphere-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .sphere-header h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #F1F5F9 !important;
        margin-bottom: 0.25rem !important;
    }
    .sphere-header p {
        color: #64748B !important;
        font-size: 0.95rem !important;
    }

    /* ── Chat messages ── */
    .chat-user {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px 12px 4px 12px;
        padding: 0.85rem 1.1rem;
        margin: 0.75rem 0;
        color: #E2E8F0 !important;
        font-size: 0.95rem;
    }
    .chat-assistant {
        background: #0F172A;
        border: 1px solid #1E3A5F;
        border-left: 3px solid #0EA5E9;
        border-radius: 4px 12px 12px 12px;
        padding: 0.85rem 1.1rem;
        margin: 0.75rem 0;
        color: #CBD5E1 !important;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .chat-assistant p, .chat-assistant li, .chat-assistant strong {
        color: #CBD5E1 !important;
    }

    /* ── Source cards ── */
    .source-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        margin: 0.35rem 0;
        font-size: 0.82rem;
        color: #94A3B8 !important;
    }
    .cat-badge {
        display: inline-block;
        background: #1E3A5F;
        color: #7DD3FC !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .score-text {
        float: right;
        color: #475569 !important;
        font-size: 0.75rem;
        font-family: monospace;
    }

    /* ── Chat input area ── */
    .stChatInput {
        background-color: #0B0F1A !important;
    }
    .stChatInput textarea {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    .stChatInput textarea::placeholder {
        color: #475569 !important;
    }
    /* Focus ring'i kapat */
    .stChatInput textarea:focus {
        border-color: #0EA5E9 !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #0B0F1A !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Fix white bar at bottom ── */
    .stBottom, .stBottom > div, [data-testid="stBottom"] {
        background-color: #0B0F1A !important;
        border-top: 1px solid #1E293B !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div {
        background-color: #0EA5E9 !important;
    }

    /* ── Toggle ── */
    .stToggle label {
        color: #CBD5E1 !important;
    }
    /* Toggle kapalı arka plan rengi */
    [data-testid="stToggle"] span[aria-checked="false"] {
        background-color: #334155 !important;
    }
    [data-testid="stToggle"] span[aria-checked="true"] {
        background-color: #0EA5E9 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #E2E8F0 !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #111827 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #0EA5E9 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.75rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F, #0369A1) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #1E293B !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background-color: #111827 !important;
        color: #94A3B8 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #0EA5E9 !important;
    }

    /* ── Fix ALL white areas ── */
    .stApp, .stApp > div, .stApp > div > div,
    .block-container, .main, .main > div,
    .stChatFloatingInputContainer,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: #0B0F1A !important;
    }

    /* ── Fix input box container ── */
    .stChatFloatingInputContainer {
        background-color: #0B0F1A !important;
        border-top: 1px solid #1E293B !important;
        padding: 0.5rem 1rem !important;
    }

    /* ── Sidebar toggle button ── */
    [data-testid="collapsedControl"] {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
    }
    /* Sidebar kapatma butonunu gizle — sidebar hep açık kalır */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {background-color: #0B0F1A !important; display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stBottom"] > div {background-color: #0B0F1A !important;}
    div[class*="StatusWidget"] {display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stAppViewContainer"] > div:first-child {background-color: #0B0F1A !important;}
</style>
""", unsafe_allow_html=True)


# ── Markdown → HTML helper ─────────────────────────────────────────────────────
def render_answer(text: str) -> str:
    return md_parser.markdown(
        text,
        extensions=["fenced_code", "tables", "nl2br"]
    )


# ── Retriever init (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_retriever():
    return Retriever(force_rebuild=False)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    top_k = st.slider(
        "Retrieved chunks (top-k)",
        min_value=1, max_value=6, value=3,
        help="Number of knowledge base entries retrieved per query."
    )

    # Fix 3: Load categories dynamically from metadata.json
    metadata_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "data", "metadata.json"
    )
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        dynamic_categories = ["All"] + sorted(
            set(item["category"] for item in _meta)
        )
    except Exception:
        dynamic_categories = ["All"]

    category_filter = st.selectbox(
        "Filter by category",
        dynamic_categories,
        help="Restrict retrieval to a specific knowledge category."
    )

    streaming = st.toggle("Streaming response", value=True,
        help="Stream the answer word-by-word as it's generated.")

    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Queries", st.session_state.total_queries)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_tokens = 0
        st.rerun()

    st.divider()
    st.markdown("""
**About**

Spheretech AI Assistant uses a RAG pipeline:

1. Query embedded with multilingual model
2. Top-k chunks retrieved from FAISS
3. Context + query sent to Groq LLM
4. Grounded answer generated

*Powered by LLaMA 3.3 70B via Groq*

---
**ℹ️ Architecture note**

This is a **Simple RAG** (stateless) implementation.
Multi-turn memory can be added via LangChain in a future phase.
    """)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sphere-header">
    <div class="sphere-badge">🛡️ AI ASSISTANT</div>
    <h1>Spheretech AI Assistant</h1>
    <p>Ask anything about Spheretech's products, services, and security solutions.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load retriever ─────────────────────────────────────────────────────────────
with st.spinner("Initialising knowledge base..."):
    try:
        retriever = load_retriever()
    except Exception as e:
        st.error(f"Failed to load retriever: {e}")
        st.stop()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user">👤 &nbsp;{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-assistant">🤖 &nbsp;{render_answer(msg["content"])}</div>',
            unsafe_allow_html=True
        )
        if msg.get("sources"):
            with st.expander("📚 Retrieved sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="source-card">
    <span class="cat-badge">{src['category']}</span>
    <span class="score-text">score: {src['score']:.3f}</span>
    <br/><br/>
    <strong style="color:#CBD5E1">{src['question']}</strong>
</div>
""", unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about Spheretech...")

if query:
    st.markdown(
        f'<div class="chat-user">👤 &nbsp;{query}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve context
    with st.spinner("Searching knowledge base..."):
        cat = None if category_filter == "All" else category_filter
        sources = retriever.get_context(query, top_k=top_k, category_filter=cat)
        context = retriever.format_context_for_prompt(sources)

    # Generate answer
    answer_placeholder = st.empty()
    full_answer = ""

    if streaming:
        answer_placeholder.markdown(
            '<div class="chat-assistant">🤖 &nbsp;▌</div>',
            unsafe_allow_html=True
        )
        for token in generate_answer_stream(query, context):
            full_answer += token
            answer_placeholder.markdown(
                f'<div class="chat-assistant">🤖 &nbsp;{full_answer}▌</div>',
                unsafe_allow_html=True
            )
        answer_placeholder.markdown(
            f'<div class="chat-assistant">🤖 &nbsp;{render_answer(full_answer)}</div>',
            unsafe_allow_html=True
        )
        tokens_used = 0
    else:
        with st.spinner("Generating answer..."):
            result = generate_answer(query, context)
        full_answer = result["answer"]
        tokens_used = result["usage"].get("total_tokens", 0)
        answer_placeholder.markdown(
            f'<div class="chat-assistant">🤖 &nbsp;{render_answer(full_answer)}</div>',
            unsafe_allow_html=True
        )

    # Show sources
    if sources:
        with st.expander("📚 Retrieved sources", expanded=False):
            for src in sources:
                st.markdown(f"""
<div class="source-card">
    <span class="cat-badge">{src['category']}</span>
    <span class="score-text">score: {src['score']:.3f}</span>
    <br/><br/>
    <strong style="color:#CBD5E1">{src['question']}</strong>
</div>
""", unsafe_allow_html=True)

    # Save to session
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
    })
    st.session_state.total_queries += 1
    st.session_state.total_tokens  += tokens_used