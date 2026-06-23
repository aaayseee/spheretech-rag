"""
embeddings.py
-------------
Loads the knowledge base CSV, builds combined text chunks
(question + answer), generates embeddings via a local
sentence-transformers model, and returns everything needed
for the vector store layer.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model ────────────────────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2  →  50+ languages including Turkish
# same 384-dim output as MiniLM-L6, runs fully local — no API key, no cost
# cross-lingual: Turkish query can match English answer chunks seamlessly
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class KnowledgeChunk:
    """Single unit of retrievable knowledge."""
    id: int
    category: str
    question: str
    answer: str
    combined_text: str   # what gets embedded
    embedding: List[float] = None


def _build_combined_text(question: str, answer: str) -> str:
    """
    Merge question + answer into one string before embedding.

    Why: embedding both surfaces semantic signal from the question
    (user intent) AND the answer (content). Retrieval accuracy
    improves significantly compared to embedding the answer alone.
    """
    return f"Question: {question}\nAnswer: {answer}"


def load_knowledge_base(csv_path: str) -> List[KnowledgeChunk]:
    """Read CSV and return a list of KnowledgeChunk objects."""
    logger.info(f"Loading knowledge base from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"id", "category", "question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    chunks = []
    for _, row in df.iterrows():
        chunk = KnowledgeChunk(
            id=int(row["id"]),
            category=str(row["category"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            combined_text=_build_combined_text(
                str(row["question"]), str(row["answer"])
            ),
        )
        chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} knowledge chunks.")
    return chunks


def generate_embeddings(chunks: List[KnowledgeChunk]) -> List[KnowledgeChunk]:
    """
    Generate a vector embedding for each chunk's combined_text.
    Embeddings are stored in-place on each KnowledgeChunk object.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk.combined_text for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks...")

    vectors = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity → dot product ready
    )

    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    logger.info("Embeddings generated successfully.")
    return chunks


def get_query_embedding(query: str) -> List[float]:
    """
    Embed a single user query at retrieval time.
    Uses the same model and normalization as the knowledge base.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector[0]


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "knowledge_base.csv"
    )

    chunks = load_knowledge_base(csv_path)
    chunks = generate_embeddings(chunks)

    # Sanity check
    print(f"\nTotal chunks embedded : {len(chunks)}")
    print(f"Embedding dimension   : {len(chunks[0].embedding)}")
    print(f"Sample combined text  :\n{chunks[0].combined_text[:200]}")
    print(f"Sample category       : {chunks[0].category}")
    print(f"Sample vector (first 5 dims): {chunks[0].embedding[:5]}")