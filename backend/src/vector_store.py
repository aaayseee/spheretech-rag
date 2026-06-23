"""
vector_store.py
---------------
Builds and manages a FAISS index over knowledge chunk embeddings.
Category and text metadata are stored alongside the index so that
retrieval returns fully enriched results — not just vector IDs.
"""

import os
import json
import numpy as np
import faiss
import logging
from typing import List, Dict, Any

from embeddings import KnowledgeChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")

EMBEDDING_DIM = 384   # paraphrase-multilingual-MiniLM-L12-v2 output size
TOP_K_DEFAULT = 3     # number of chunks to retrieve per query


# ── Build ─────────────────────────────────────────────────────────────────────

def build_index(chunks: List[KnowledgeChunk]) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (inner product) index from chunk embeddings.

    Why IndexFlatIP?
    - Embeddings are L2-normalised in embeddings.py so inner product == cosine similarity.
    - Exact search (no approximation) — suitable for our dataset size (50 chunks).
    - For 10 000+ chunks, switch to IndexIVFFlat for speed.
    """
    logger.info("Building FAISS index...")

    vectors = np.array(
        [chunk.embedding for chunk in chunks], dtype=np.float32
    )

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    logger.info(f"FAISS index built — {index.ntotal} vectors stored.")
    return index


def save_index(index: faiss.Index, chunks: List[KnowledgeChunk]) -> None:
    """Persist the FAISS index and metadata to disk."""
    faiss.write_index(index, INDEX_PATH)
    logger.info(f"FAISS index saved → {INDEX_PATH}")

    # Metadata: everything except the raw embedding vector
    metadata = [
        {
            "id":            chunk.id,
            "category":      chunk.category,
            "question":      chunk.question,
            "answer":        chunk.answer,
            "combined_text": chunk.combined_text,
        }
        for chunk in chunks
    ]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Metadata saved → {METADATA_PATH}")


def load_index() -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load a previously saved FAISS index and its metadata from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Index or metadata not found. Run build_and_save_index() first."
        )

    index = faiss.read_index(INDEX_PATH)
    logger.info(f"FAISS index loaded — {index.ntotal} vectors.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Metadata loaded — {len(metadata)} records.")
    return index, metadata


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(
    query_vector: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int = TOP_K_DEFAULT,
    category_filter: str = None,
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for the top-k most similar chunks.

    Parameters
    ----------
    query_vector    : 1-D numpy array, same dim as stored embeddings
    index           : loaded FAISS index
    metadata        : list of dicts parallel to the index vectors
    top_k           : number of results to return
    category_filter : optional — restrict results to a specific category

    Returns
    -------
    List of metadata dicts enriched with a `score` field (cosine similarity).
    """
    query = np.array([query_vector], dtype=np.float32)

    # Retrieve more candidates when filtering so we still get top_k after
    fetch_k = top_k * 4 if category_filter else top_k

    scores, indices = index.search(query, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_meta = metadata[idx].copy()
        chunk_meta["score"] = float(score)

        if category_filter and chunk_meta["category"] != category_filter:
            continue

        results.append(chunk_meta)

        if len(results) == top_k:
            break

    logger.info(
        f"Retrieved {len(results)} chunks "
        f"(filter={category_filter or 'none'}, top_k={top_k})"
    )
    return results


# ── Convenience wrapper ───────────────────────────────────────────────────────

def build_and_save_index(chunks: List[KnowledgeChunk]) -> None:
    """One-shot: build FAISS index from chunks and persist everything."""
    index = build_index(chunks)
    save_index(index, chunks)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from embeddings import load_knowledge_base

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "knowledge_base.csv"
    )

    # Use random embeddings so the test runs without network access
    chunks = load_knowledge_base(csv_path)
    for chunk in chunks:
        vec = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)   # normalise
        chunk.embedding = vec

    build_and_save_index(chunks)

    # Reload and query
    index, metadata = load_index()
    query_vec = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    results = retrieve(query_vec, index, metadata, top_k=3)

    print(f"\n✅ Index size          : {index.ntotal}")
    print(f"✅ Metadata records    : {len(metadata)}")
    print(f"✅ Retrieved chunks    : {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.4f} | category={r['category']}")
        print(f"       question: {r['question'][:80]}")

    # Category filter test
    filtered = retrieve(
        query_vec, index, metadata,
        top_k=2, category_filter="Products"
    )
    print(f"\nCategory filter (Products) returned: {len(filtered)} chunks")