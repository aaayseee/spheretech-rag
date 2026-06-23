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
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background-color: #0B0F1A;
        color: #E2E8F0;
    }

    /* ── Header ── */
    .sphere-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .sphere-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .sphere-header p {
        color: #64748B;
        font-size: 0.95rem;
    }
    .sphere-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1E3A5F, #0EA5E9);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.75rem;
    }

    /* ── Chat messages ── */
    .chat-user {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px 12px 4px 12px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0;
        color: #E2E8F0;
        font-size: 0.95rem;
    }
    .chat-assistant {
        background: #0F172A;
        border: 1px solid #1E3A5F;
        border-left: 3px solid #0EA5E9;
        border-radius: 4px 12px 12px 12px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0;
        color: #CBD5E1;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* ── Sources panel ── */
    .source-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        margin: 0.35rem 0;
        font-size: 0.82rem;
        color: #94A3B8;
    }
    .source-card .cat-badge {
        display: inline-block;
        background: #1E3A5F;
        color: #7DD3FC;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 1px 7px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .source-card .score {
        float: right;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94A3B8;
        font-size: 0.85rem;
    }

    /* ── Input box ── */
    .stTextInput input, .stChatInput textarea {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F1F5F9 !important;
        border-radius: 8px !important;
    }

    /* ── Buttons ── */
    .stButton button {
        background: linear-gradient(135deg, #1E3A5F, #0369A1);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
        transition: opacity 0.2s;
    }
    .stButton button:hover {
        opacity: 0.85;
    }

    /* ── Divider ── */
    hr {
        border-color: #1E293B;
        margin: 1.25rem 0;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
    }
    [data-testid="metric-container"] label {
        color: #64748B !important;
        font-size: 0.75rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0EA5E9 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    /* ── Hide streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Markdown → HTML helper (Fix 1) ───────────────────────────────────────────
def render_answer(text: str) -> str:
    """
    Convert LLM markdown output to safe HTML so bullet points,
    bold text and code blocks render correctly inside custom divs.
    """
    return md_parser.markdown(
        text,
        extensions=["fenced_code", "tables", "nl2br"]
    )


# ── Retriever init (cached — loads once per session) ───────────────────────────
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
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Tokens", st.session_state.total_tokens)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_tokens = 0
        st.rerun()

    st.divider()
    st.markdown("""
**About**

Spheretech AI Assistant uses a RAG (Retrieval-Augmented Generation) pipeline:

1. Your question is embedded with a multilingual model
2. The top-k most relevant knowledge chunks are retrieved from FAISS
3. Retrieved context + your question are sent to Groq LLM
4. A grounded answer is generated

*Powered by LLaMA 3.3 70B via Groq*

---
**ℹ️ Architecture note**

This is a **Simple RAG** (stateless) implementation. Chat history is displayed
for UX convenience but only the latest question is sent to the LLM.
Multi-turn memory can be added via LangChain in a future phase.
    """)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sphere-header">
    <div class="sphere-badge">🛡️ AI Assistant</div>
    <h1>Spheretech AI Assistant</h1>
    <p>Ask anything about Spheretech's products, services, and security solutions.</p>
</div>
""", unsafe_allow_html=True)

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
            f'<div class="chat-user">👤 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-assistant">🤖 {render_answer(msg["content"])}</div>',
            unsafe_allow_html=True
        )
        if msg.get("sources"):
            with st.expander("📚 Retrieved sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="source-card">
    <span class="cat-badge">{src['category']}</span>
    <span class="score">score: {src['score']:.3f}</span>
    <br/>
    <strong>{src['question']}</strong>
</div>
""", unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about Spheretech...")

if query:
    # Display user message
    st.markdown(
        f'<div class="chat-user">👤 {query}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": query})

    # ── Retrieve context ──
    with st.spinner("Searching knowledge base..."):
        cat = None if category_filter == "All" else category_filter
        sources = retriever.get_context(query, top_k=top_k, category_filter=cat)
        context = retriever.format_context_for_prompt(sources)

    # ── Generate answer ──
    answer_placeholder = st.empty()
    full_answer = ""

    if streaming:
        with st.spinner(""):
            answer_placeholder.markdown(
                '<div class="chat-assistant">🤖 ▌</div>',
                unsafe_allow_html=True
            )
            for token in generate_answer_stream(query, context):
                full_answer += token
                answer_placeholder.markdown(
                    f'<div class="chat-assistant">🤖 {full_answer}▌</div>',
                    unsafe_allow_html=True
                )
            answer_placeholder.markdown(
                f'<div class="chat-assistant">🤖 {render_answer(full_answer)}</div>',
                unsafe_allow_html=True
            )
        tokens_used = 0   # streaming doesn't return usage stats
    else:
        with st.spinner("Generating answer..."):
            result = generate_answer(query, context)

        if result["success"]:
            full_answer = result["answer"]
            tokens_used = result["usage"].get("total_tokens", 0)
        else:
            full_answer = result["answer"]
            tokens_used = 0

        answer_placeholder.markdown(
            f'<div class="chat-assistant">🤖 {render_answer(full_answer)}</div>',
            unsafe_allow_html=True
        )

    # ── Show sources ──
    if sources:
        with st.expander("📚 Retrieved sources", expanded=False):
            for src in sources:
                st.markdown(f"""
<div class="source-card">
    <span class="cat-badge">{src['category']}</span>
    <span class="score">score: {src['score']:.3f}</span>
    <br/>
    <strong>{src['question']}</strong>
</div>
""", unsafe_allow_html=True)

    # ── Save to session ──
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
    })
    st.session_state.total_queries += 1
    st.session_state.total_tokens  += tokens_used