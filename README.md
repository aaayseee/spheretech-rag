# 🛡️ Spheretech AI Assistant — RAG Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** system built for Spheretech, a cybersecurity software company. The assistant answers questions about Spheretech's products, services, and policies by retrieving relevant knowledge and generating grounded responses via a Large Language Model.

🌐 **Live Demo:** [spheretech-rag.streamlit.app](https://spheretech-rag.streamlit.app)

---

## 📌 Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Knowledge Base](#knowledge-base)
- [Design Decisions](#design-decisions)
- [Bonus Features](#bonus-features)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│                    (app.py)                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              RAG Pipeline (backend/src)              │
│                                                     │
│  1. embeddings.py   →  Embed query                  │
│     paraphrase-multilingual-MiniLM-L12-v2           │
│     (50+ languages including Turkish)               │
│                                                     │
│  2. vector_store.py →  FAISS similarity search      │
│     IndexFlatIP + category metadata                 │
│                                                     │
│  3. retriever.py    →  Top-k context retrieval      │
│     score threshold + category filter               │
│                                                     │
│  4. llm_client.py   →  Groq LLM generation         │
│     llama-3.3-70b-versatile                         │
│     standard + streaming                            │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           FastAPI Backend (bonus)                   │
│   /health  /categories  /retrieve  /ask  /ask/stream│
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) |
| Vector Store | FAISS `IndexFlatIP` |
| LLM | LLaMA 3.3 70B via Groq API |
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
spheretech-rag/
├── backend/
│   ├── src/
│   │   ├── embeddings.py      # HuggingFace multilingual embeddings
│   │   ├── vector_store.py    # FAISS index build, save, load, retrieve
│   │   ├── retriever.py       # Retriever class — pipeline orchestrator
│   │   └── llm_client.py      # Groq API — standard + streaming
│   ├── data/
│   │   └── knowledge_base.csv # 50 Q&A chunks across 19 categories
│   ├── main.py                # FastAPI server (bonus)
│   └── requirements.txt
├── frontend/
│   ├── app.py                 # Streamlit UI
│   └── requirements.txt
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.example
├── .gitignore
└── .dockerignore
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/aaayseee/spheretech-rag.git
cd spheretech-rag
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Install dependencies

```bash
pip install -r frontend/requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First run:** The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, ~500MB) will be downloaded automatically and cached locally. Subsequent runs are instant.

---

## Running with Docker

### Start everything

```bash
# Copy and fill in your API key first
cp .env.example .env

# Build and start
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI (bonus) | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

### Start frontend only (Simple RAG mode)

```bash
docker-compose up --build frontend
```

### Stop

```bash
docker-compose down
```

---

## Streamlit Cloud Deploy

The app is deployed at [spheretech-rag.streamlit.app](https://spheretech-rag.streamlit.app).

To deploy your own instance:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → Main file: `frontend/app.py`
4. Add `GROQ_API_KEY = "your_key"` under Secrets
5. Deploy!

---

## API Reference

The FastAPI backend exposes the following endpoints:

### `GET /health`
Liveness check for Docker healthcheck.

```json
{ "status": "ok", "service": "spheretech-rag-backend" }
```

### `GET /categories`
Returns all unique categories in the knowledge base.

```json
{ "categories": ["Company Overview", "Products", ...], "total": 19 }
```

### `POST /retrieve`
Retrieve top-k relevant chunks without LLM generation.

```json
// Request
{
  "query": "What is SphereShield?",
  "top_k": 3,
  "category_filter": "Products",
  "score_threshold": 0.0
}

// Response
{
  "query": "What is SphereShield?",
  "chunks": [{ "id": 5, "category": "Products", "question": "...", "answer": "...", "score": 0.91 }],
  "total": 3
}
```

### `POST /ask`
Full RAG pipeline — retrieval + LLM generation.

```json
// Request
{ "query": "How much does SphereShield cost?", "top_k": 3 }

// Response
{
  "query": "...",
  "answer": "SphereShield is licensed at $15 per endpoint per month...",
  "model": "llama-3.3-70b-versatile",
  "chunks_used": [...],
  "usage": { "prompt_tokens": 412, "completion_tokens": 118 },
  "success": true
}
```

### `POST /ask/stream`
Same as `/ask` but returns a streaming `text/plain` response (SSE).

---

## Knowledge Base

The knowledge base (`backend/data/knowledge_base.csv`) contains **50 entries** across **19 categories**:

| Category | Examples |
|---|---|
| Products | SphereShield, SphereSOC, ThreatRadar, SecureVault |
| Services | MDR, penetration testing, incident response |
| Compliance | ISO 27001, GDPR, KVKK, SOC 2 |
| Pricing | Per-endpoint pricing, free trial, payment terms |
| Technical Support | SLA tiers, response times |
| Edge Cases (Turkish) | "Fiyatlarınız nedir?", "Siber saldırıya uğradık" |
| Edge Cases (Negative) | Complaint-style questions |
| Edge Cases (Cross-category) | KVKK + product, GDPR + cloud |

Each entry is embedded as `"Question: {q}\nAnswer: {a}"` — combining both fields significantly improves retrieval accuracy compared to embedding the answer alone.

---

## Design Decisions

### Why `paraphrase-multilingual-MiniLM-L12-v2`?
The knowledge base contains Turkish edge-case entries. This model supports 50+ languages and enables cross-lingual retrieval — a Turkish query can match an English answer chunk.

### Why `IndexFlatIP` (FAISS)?
Embeddings are L2-normalised, so inner product equals cosine similarity. Exact search is appropriate for 50 chunks. For 10,000+ chunks, `IndexIVFFlat` would be preferred.

### Why `temperature=0.2`?
Low temperature keeps the LLM factual and consistent — critical for a customer-facing assistant where hallucinations damage trust.

### Stateless RAG (Simple RAG)
The LLM receives only the current query — no conversation history. Chat history is displayed in the UI for UX purposes. Multi-turn memory (e.g. via LangChain) can be added in a future phase.

### Why separate `Dockerfile.backend` and `Dockerfile.frontend`?
Clean separation of concerns. In the Simple RAG phase the frontend imports backend modules directly. When the FastAPI bonus is enabled, only the API URL needs to change in `app.py` — zero refactoring.

---

## Bonus Features

| Feature | Status | Location |
|---|---|---|
| FastAPI backend separation | ✅ | `backend/main.py` |
| Streaming LLM response | ✅ | `llm_client.py` → `generate_answer_stream()` |
| Evaluation methodology | 📋 Proposed | See below |

### Evaluation Methodology

To measure retrieval and generation quality, the following approach is proposed:

**Retrieval evaluation:**
- Create a ground-truth dataset: `(question, expected_category)` pairs
- Metric: **Hit Rate @k** — does the correct chunk appear in the top-k results?
- Metric: **MRR** (Mean Reciprocal Rank)

**Generation evaluation:**
- Create a ground-truth dataset: `(question, expected_answer)`
- Metric: **ROUGE-L** for answer overlap
- Metric: **BERTScore** for semantic similarity
- Metric: **Faithfulness** — does the answer stay within the retrieved context?

```python
# Example evaluation snippet
ground_truth = [
    {"query": "What is SphereShield?", "expected_category": "Products"},
    {"query": "Fiyatlarınız nedir?",   "expected_category": "Pricing and Licensing"},
]

hit_count = 0
for item in ground_truth:
    results = retriever.get_context(item["query"], top_k=3)
    if any(r["category"] == item["expected_category"] for r in results):
        hit_count += 1

hit_rate = hit_count / len(ground_truth)
print(f"Hit Rate @3: {hit_rate:.2%}")
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Free API key from [console.groq.com](https://console.groq.com) |

---

*Built as part of the Spheretech Case Study — API-Driven AI Assistant (Simple RAG Pipeline)*