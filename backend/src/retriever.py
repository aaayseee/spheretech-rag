"""
retriever.py
------------
Orchestrates the full retrieval pipeline:
  1. Embed the user query (multilingual MiniLM)
  2. Search the FAISS index for top-k similar chunks
  3. Return enriched context ready for the LLM prompt

This module is the single entry point for both the Streamlit UI
and the FastAPI backend — neither needs to know about FAISS or
embeddings directly.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))

import embeddings as emb_module
from embeddings import generate_embeddings, load_knowledge_base
from vector_store import (
    build_and_save_index,
    load_index,
    retrieve,
    INDEX_PATH,
    METADATA_PATH,
    TOP_K_DEFAULT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.csv")


# ── Index initialisation ──────────────────────────────────────────────────────

def initialise_index(force_rebuild: bool = False) -> tuple:
    """
    Load FAISS index from disk if it exists, otherwise build it from scratch.

    Parameters
    ----------
    force_rebuild : if True, always re-embed and rebuild even if index exists.

    Returns
    -------
    (index, metadata) tuple ready for retrieval.
    """
    index_exists = (
        os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)
    )

    if index_exists and not force_rebuild:
        logger.info("Existing index found — loading from disk.")
        return load_index()

    logger.info("Building index from scratch...")
    chunks = load_knowledge_base(CSV_PATH)
    chunks = generate_embeddings(chunks)
    build_and_save_index(chunks)
    return load_index()


# ── Core retriever ────────────────────────────────────────────────────────────

class Retriever:
    """
    Stateful retriever that holds the FAISS index in memory
    for fast repeated queries.

    Usage
    -----
        retriever = Retriever()
        results   = retriever.get_context("What is SphereShield?")
    """

    def __init__(self, force_rebuild: bool = False):
        logger.info("Initialising Retriever...")
        self.index, self.metadata = initialise_index(force_rebuild)
        logger.info("Retriever ready.")

    def get_context(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant knowledge chunks for a user query.

        Parameters
        ----------
        query            : raw user question (any supported language)
        top_k            : number of chunks to return
        category_filter  : optional category name to restrict results
        score_threshold  : minimum cosine similarity score (0.0 = no filter)

        Returns
        -------
        List of dicts with keys: id, category, question, answer, score
        """
        logger.info(f"Query received: '{query}'")

        # Embed the query with the same multilingual model
        # Called via module reference so it can be mocked in tests
        query_vector = emb_module.get_query_embedding(query)

        # Search FAISS
        results = retrieve(
            query_vector=query_vector,
            index=self.index,
            metadata=self.metadata,
            top_k=top_k,
            category_filter=category_filter,
        )

        # Optional score filtering
        if score_threshold > 0.0:
            before = len(results)
            results = [r for r in results if r["score"] >= score_threshold]
            logger.info(
                f"Score threshold ({score_threshold}) applied: "
                f"{before} → {len(results)} chunks"
            )

        return results

    def format_context_for_prompt(
        self, results: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved chunks into a clean context block
        ready to be injected into the LLM prompt.

        Example output
        --------------
        [1] Category: Products
            Q: What is SphereShield?
            A: SphereShield is Spheretech's flagship EDR platform...

        [2] Category: Pricing and Licensing
            Q: How is SphereShield priced?
            A: SphereShield is licensed on a per-endpoint basis...
        """
        if not results:
            return "No relevant information found in the knowledge base."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] Category: {r['category']}")
            lines.append(f"    Q: {r['question']}")
            lines.append(f"    A: {r['answer']}")
            lines.append(f"    Relevance score: {r['score']:.4f}")
            lines.append("")

        return "\n".join(lines).strip()


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import embeddings as emb_module

    # ── 1. Pre-build index with random embeddings (no network needed) ──
    chunks = load_knowledge_base(CSV_PATH)
    for chunk in chunks:
        vec = np.random.rand(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        chunk.embedding = vec
    build_and_save_index(chunks)

    # ── 2. Mock get_query_embedding so Retriever works without HuggingFace ──
    def _mock_query_embedding(query: str) -> np.ndarray:
        """Returns a reproducible random vector for testing."""
        rng = np.random.default_rng(seed=abs(hash(query)) % (2**32))
        vec = rng.random(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    emb_module.get_query_embedding = _mock_query_embedding

    # ── 3. Instantiate and test the Retriever class ──
    retriever = Retriever(force_rebuild=False)

    test_queries = [
        "What is SphereShield?",
        "Fiyatlarınız nedir?",
        "Do you support cloud environments?",
    ]

    for query in test_queries:
        print(f"\n{'─'*55}")
        print(f"Query: {query}")

        # Test get_context()
        results = retriever.get_context(query, top_k=3)
        assert len(results) == 3, "Expected 3 results"
        print(f"   get_context()               → {len(results)} chunks returned")
        print(f"   Top result category        : {results[0]['category']}")
        print(f"   Top result score           : {results[0]['score']:.4f}")
        print(f"   Metadata keys              : {list(results[0].keys())}")

        # Test format_context_for_prompt()
        context_block = retriever.format_context_for_prompt(results)
        assert "[1]" in context_block and "[2]" in context_block
        print(f"✅ format_context_for_prompt() → {len(context_block)} chars")

    # ── 4. Edge case: empty results ──
    empty_block = retriever.format_context_for_prompt([])
    assert empty_block == "No relevant information found in the knowledge base."
    print(f"\nEmpty result handling        → OK")

    # ── 5. Edge case: score_threshold filters everything out ──
    high_threshold = retriever.get_context(
        "SphereShield pricing", top_k=3, score_threshold=0.9999
    )
    assert len(high_threshold) == 0
    print(f"score_threshold=0.9999       → {len(high_threshold)} chunks (expected 0)")

    # ── 6. Edge case: category_filter ──
    filtered = retriever.get_context(
        "security product", top_k=5, category_filter="Products"
    )
    assert all(r["category"] == "Products" for r in filtered)
    print(f"category_filter='Products'   → {len(filtered)} chunks, all correct category")

    print(f"\n{'─'*55}")
    print("All Retriever methods verified successfully.")