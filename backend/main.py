"""
main.py
-------
FastAPI backend for the Spheretech AI Assistant.
Wraps the RAG pipeline (retriever + LLM) in REST API endpoints
so the Streamlit frontend can call it over HTTP instead of
importing backend modules directly.

Endpoints:
    GET  /health          →  liveness check
    GET  /categories      →  list all knowledge base categories
    POST /retrieve        →  retrieve top-k chunks for a query
    POST /ask             →  full RAG: retrieve + generate answer
    POST /ask/stream      →  full RAG with streaming response

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from retriever import Retriever
from llm_client import generate_answer, generate_answer_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.
    Using lifespan instead of @app.on_event("startup") which is
    deprecated in FastAPI v0.93.0+.
    """
    # Startup: load retriever once into app state
    logger.info("Loading retriever on startup...")
    app.state.retriever = Retriever(force_rebuild=False)
    logger.info("Retriever ready.")
    yield
    # Shutdown: cleanup if needed (placeholder for future resource release)
    logger.info("Shutting down Spheretech API.")


# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spheretech AI Assistant API",
    description="RAG pipeline API — retrieval + LLM generation over Spheretech knowledge base.",
    version="1.0.0",
    lifespan=lifespan,      # modern startup/shutdown management
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allows Streamlit frontend (localhost:8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       example="What is SphereShield?")
    top_k: int = Field(default=3, ge=1, le=10,
                       description="Number of chunks to retrieve")
    category_filter: Optional[str] = Field(default=None,
                       example="Products",
                       description="Optional category to restrict retrieval")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0,
                       description="Minimum cosine similarity score")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       example="How much does SphereShield cost?")
    top_k: int = Field(default=3, ge=1, le=10)
    category_filter: Optional[str] = Field(default=None)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkResponse(BaseModel):
    id: int
    category: str
    question: str
    answer: str
    score: float


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[ChunkResponse]
    total: int


class AskResponse(BaseModel):
    query: str
    answer: str
    model: str
    chunks_used: list[ChunkResponse]
    usage: dict
    success: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness check — used by Docker healthcheck."""
    return {"status": "ok", "service": "spheretech-rag-backend"}


@app.get("/categories", tags=["Knowledge Base"])
async def get_categories():
    """
    Return all unique categories present in the knowledge base.
    Used by the Streamlit frontend to populate the category filter dropdown.
    """
    metadata_path = os.path.join(
        os.path.dirname(__file__), "data", "metadata.json"
    )
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        categories = sorted(set(item["category"] for item in metadata))
        return {"categories": categories, "total": len(categories)}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base index not found. Please rebuild the index."
        )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["RAG Pipeline"])
async def retrieve(request: RetrieveRequest):
    """
    Retrieve the most relevant knowledge chunks for a given query.
    Returns raw chunks without LLM generation — useful for debugging retrieval quality.
    """
    retriever: Retriever = app.state.retriever

    try:
        chunks = retriever.get_context(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
            score_threshold=request.score_threshold,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    return RetrieveResponse(
        query=request.query,
        chunks=[ChunkResponse(**{k: v for k, v in c.items()
                                 if k in ChunkResponse.model_fields})
                for c in chunks],
        total=len(chunks),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG Pipeline"])
async def ask(request: AskRequest):
    """
    Full RAG pipeline: retrieve relevant chunks + generate an LLM answer.
    Returns the complete answer with source chunks and token usage.
    """
    retriever: Retriever = app.state.retriever

    # Step 1: Retrieve context
    try:
        chunks = retriever.get_context(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
            score_threshold=request.score_threshold,
        )
        context = retriever.format_context_for_prompt(chunks)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Step 2: Generate answer
    result = generate_answer(query=request.query, context=context)

    return AskResponse(
        query=request.query,
        answer=result["answer"],
        model=result["model"],
        chunks_used=[ChunkResponse(**{k: v for k, v in c.items()
                                      if k in ChunkResponse.model_fields})
                     for c in chunks],
        usage=result.get("usage", {}),
        success=result["success"],
    )


@app.post("/ask/stream", tags=["RAG Pipeline"])
async def ask_stream(request: AskRequest):
    """
    Full RAG pipeline with streaming response.
    Returns a Server-Sent Events (SSE) stream — text appears word by word.

    Streamlit usage:
        response = requests.post("/ask/stream", json=..., stream=True)
        for line in response.iter_lines():
            print(line.decode())
    """
    retriever: Retriever = app.state.retriever

    # Retrieve context first (not streamed)
    try:
        chunks = retriever.get_context(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter,
        )
        context = retriever.format_context_for_prompt(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    def token_generator():
        for token in generate_answer_stream(
            query=request.query, context=context
        ):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )


# ── Dev entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)