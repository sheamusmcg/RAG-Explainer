"""Session state initialization, predicates, and downstream clearing."""

import streamlit as st

_DEFAULTS = {
    # Documents
    "documents": None,          # list of dicts: [{title, text}]
    "doc_source": None,         # "sample" or "custom"
    # Chunks
    "chunks": None,             # list of dicts: [{text, source_title, chunk_index}]
    "chunk_size": 500,
    "chunk_overlap": 50,
    # Embeddings
    "embeddings": None,         # numpy array (n_chunks, dim)
    "embedding_method": "tfidf",# "tfidf" or "transformer"
    "token_counts": None,       # list of ints
    # Index
    "faiss_index": None,        # faiss.IndexFlatL2
    # Retrieval
    "query_text": None,
    "query_embedding": None,    # numpy array (1, dim)
    "retrieval_results": None,  # list of dicts
    "top_k": 3,
    # Generation
    "assembled_prompt": None,
    "generated_response": None,
    # Evaluation
    "eval_scores": None,
}

# Maps a stage to every session-state key that should be reset when that stage
# is re-run (i.e. everything *downstream* of it).
_DOWNSTREAM = {
    "documents": [
        "chunks", "embeddings", "token_counts", "faiss_index",
        "query_text", "query_embedding", "retrieval_results",
        "assembled_prompt", "generated_response", "eval_scores",
    ],
    "chunks": [
        "embeddings", "token_counts", "faiss_index",
        "query_text", "query_embedding", "retrieval_results",
        "assembled_prompt", "generated_response", "eval_scores",
    ],
    "embeddings": [
        "faiss_index",
        "query_text", "query_embedding", "retrieval_results",
        "assembled_prompt", "generated_response", "eval_scores",
    ],
    "index": [
        "query_text", "query_embedding", "retrieval_results",
        "assembled_prompt", "generated_response", "eval_scores",
    ],
    "retrieval": [
        "assembled_prompt", "generated_response", "eval_scores",
    ],
    "generation": [
        "eval_scores",
    ],
}


def init_state():
    """Ensure every expected key exists in session state (first run only)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_downstream(from_stage: str):
    """Reset all session-state keys downstream of *from_stage* to None."""
    for key in _DOWNSTREAM.get(from_stage, []):
        st.session_state[key] = None


# ── Predicates ───────────────────────────────────────────────────────────

def has_documents() -> bool:
    return st.session_state.get("documents") is not None


def has_chunks() -> bool:
    return st.session_state.get("chunks") is not None


def has_embeddings() -> bool:
    return st.session_state.get("embeddings") is not None


def has_index() -> bool:
    return st.session_state.get("faiss_index") is not None


def has_retrieval_results() -> bool:
    return st.session_state.get("retrieval_results") is not None


def has_assembled_prompt() -> bool:
    return st.session_state.get("assembled_prompt") is not None


def has_generated_response() -> bool:
    return st.session_state.get("generated_response") is not None
