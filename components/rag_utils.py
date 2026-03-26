"""Core RAG utility functions: chunking, embedding, indexing, retrieval, evaluation."""

from __future__ import annotations

import re

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping chunks by character count."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Chunk every document, preserving source metadata.

    Returns a list of dicts: {text, source_title, chunk_index}.
    """
    all_chunks: list[dict] = []
    for doc in documents:
        parts = chunk_text(doc["text"], chunk_size, overlap)
        for i, part in enumerate(parts):
            all_chunks.append(
                {
                    "text": part,
                    "source_title": doc["title"],
                    "chunk_index": i,
                }
            )
    return all_chunks


# ── Tokenization ─────────────────────────────────────────────────────────

def count_tokens(texts: list[str], encoding_name: str = "cl100k_base") -> list[int]:
    """Return the token count for each text using tiktoken."""
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return [len(enc.encode(t)) for t in texts]


# ── Embeddings ───────────────────────────────────────────────────────────

# We support two embedding modes:
#   "tfidf"  — fast, no downloads, uses scikit-learn (default)
#   "transformer" — better quality, requires sentence-transformers + torch

@st.cache_resource(show_spinner=False)
def _build_tfidf_model(corpus_key: str, texts: list[str]):
    """Fit a TF-IDF vectorizer on the corpus. Cached by corpus_key."""
    vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
    vectorizer.fit(texts)
    return vectorizer


def embed_texts_tfidf(texts: list[str], fit_texts: list[str] | None = None) -> np.ndarray:
    """Embed texts using TF-IDF.

    If *fit_texts* is provided, the vectorizer is fit on those texts first
    (used so the query is projected into the same space as the corpus).
    """
    corpus = fit_texts if fit_texts is not None else texts
    corpus_key = str(hash(tuple(corpus)))
    vectorizer = _build_tfidf_model(corpus_key, corpus)
    # Convert sparse → dense for FAISS compatibility
    return vectorizer.transform(texts).toarray().astype(np.float32)


@st.cache_resource(show_spinner=False)
def _load_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load a sentence-transformers model (cached across reruns)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts_transformer(texts: list[str]) -> np.ndarray:
    """Embed texts using sentence-transformers."""
    model = _load_transformer_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def embed_texts(
    texts: list[str],
    method: str = "tfidf",
    fit_texts: list[str] | None = None,
) -> np.ndarray:
    """Unified embedding function.

    method: "tfidf" or "transformer"
    fit_texts: for tfidf, the corpus to fit the vectorizer on
    """
    if method == "transformer":
        return embed_texts_transformer(texts)
    else:
        return embed_texts_tfidf(texts, fit_texts=fit_texts)


# ── Dimensionality Reduction ─────────────────────────────────────────────

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    method: str = "pca",
) -> np.ndarray:
    """Project embeddings down to 2-D or 3-D for plotting."""
    # Ensure n_components doesn't exceed features or samples
    max_components = min(embeddings.shape[0], embeddings.shape[1])
    n_components = min(n_components, max_components)
    if n_components < 2:
        # Pad with zeros if we can't get 2 components
        reduced = embeddings[:, :1] if embeddings.shape[1] >= 1 else np.zeros((embeddings.shape[0], 1))
        return np.hstack([reduced, np.zeros((embeddings.shape[0], 2 - reduced.shape[1]))])

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:  # tsne
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, embeddings.shape[0] - 1))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return reducer.fit_transform(embeddings)


# ── FAISS Index ──────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS Flat-L2 index from *embeddings*."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def search_index(
    index,
    query_embedding: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index. Returns (distances, indices)."""
    query = query_embedding.astype(np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances, indices = index.search(query, top_k)
    return distances[0], indices[0]


def l2_to_similarity(distances: np.ndarray) -> np.ndarray:
    """Convert L2 distances to a 0-1 similarity score."""
    return 1.0 / (1.0 + distances)


# ── Prompt Assembly ──────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based ONLY on the "
    "provided context. If the context doesn't contain enough information to answer, "
    "say so. Do not make up information."
)


def assemble_prompt(
    query: str,
    chunks: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Build the full RAG prompt from system prompt + context + query."""
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION:\n{query}"
    )


# ── Evaluation Helpers ───────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.flatten().reshape(1, -1)
    b = b.flatten().reshape(1, -1)
    return float(sklearn_cosine(a, b)[0, 0])


def compute_context_relevance(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> tuple[float, list[float]]:
    """Average cosine similarity between query and each retrieved chunk.

    Returns (overall_score, per_chunk_scores).
    """
    scores = [
        cosine_similarity(query_embedding, chunk_embeddings[i])
        for i in range(chunk_embeddings.shape[0])
    ]
    return float(np.mean(scores)), scores


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.split()) >= 3]


def _word_overlap(sent: str, chunk: str) -> float:
    """Fraction of words in *sent* that appear in *chunk*."""
    sent_words = set(sent.lower().split())
    chunk_words = set(chunk.lower().split())
    if not sent_words:
        return 0.0
    return len(sent_words & chunk_words) / len(sent_words)


def compute_faithfulness(response: str, chunks: list[str]) -> float:
    """Heuristic faithfulness: fraction of response sentences that have
    high word overlap with at least one retrieved chunk."""
    sentences = _split_sentences(response)
    if not sentences:
        return 0.0
    grounded_count = 0
    for sent in sentences:
        max_overlap = max(_word_overlap(sent, c) for c in chunks)
        if max_overlap >= 0.4:
            grounded_count += 1
    return grounded_count / len(sentences)


def compute_answer_relevance(
    query_embedding: np.ndarray,
    response_embedding: np.ndarray,
) -> float:
    """Cosine similarity between query and response embeddings."""
    return cosine_similarity(query_embedding, response_embedding)
