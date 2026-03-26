import streamlit as st
import pandas as pd
from components.state_manager import has_documents, has_chunks, clear_downstream
from components.rag_utils import chunk_documents

# ── Prerequisite ─────────────────────────────────────────────────────────
if not has_documents():
    st.title("Chunking")
    st.warning("Please load documents first.")
    st.page_link("pages/02_load_documents.py", label="Go to Load Documents", icon=":material/arrow_back:")
    st.stop()

st.title("Chunking")
st.write(
    "Documents are too long to embed and search as-is. We split them into smaller "
    "overlapping **chunks** so each piece can be independently embedded and retrieved."
)

# ── Controls ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    chunk_size = st.slider(
        "Chunk size (characters)", 100, 2000, st.session_state.get("chunk_size", 500), step=50,
        help="How many characters per chunk. Smaller = more precise retrieval; larger = more context per chunk.",
    )
with col2:
    chunk_overlap = st.slider(
        "Overlap (characters)", 0, 500, st.session_state.get("chunk_overlap", 50), step=25,
        help="How many characters overlap between adjacent chunks. Overlap prevents losing context at boundaries.",
    )

if st.button("Chunk Documents", type="primary"):
    docs = st.session_state["documents"]
    with st.spinner("Chunking documents..."):
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    clear_downstream("chunks")
    st.session_state["chunks"] = chunks
    st.session_state["chunk_size"] = chunk_size
    st.session_state["chunk_overlap"] = chunk_overlap
    st.success(f"Created **{len(chunks)}** chunks.")

# ── Results ──────────────────────────────────────────────────────────────
if has_chunks():
    chunks = st.session_state["chunks"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        avg_len = int(sum(len(c["text"]) for c in chunks) / len(chunks))
        st.metric("Avg Length (chars)", avg_len)
    with col3:
        avg_words = int(sum(len(c["text"].split()) for c in chunks) / len(chunks))
        st.metric("Avg Length (words)", avg_words)

    chunk_df = pd.DataFrame(
        {
            "#": range(1, len(chunks) + 1),
            "Source": [c["source_title"] for c in chunks],
            "Characters": [len(c["text"]) for c in chunks],
            "Preview": [c["text"][:100] + "..." for c in chunks],
        }
    )
    st.dataframe(chunk_df, use_container_width=True, hide_index=True)

    with st.expander("View full chunk texts"):
        for i, c in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}** (from *{c['source_title']}*)")
            st.text(c["text"])
            st.divider()

    # ── Learn more ───────────────────────────────────────────────────────
    with st.expander("Learn more: Chunk Size Tradeoffs"):
        st.write(
            "**Too small** — The LLM can't understand the retrieved fragment because "
            "it's been stripped of surrounding context.\n\n"
            "**Too large** — Expensive to process, dilutes similarity search, and the "
            "relevant nugget is buried in noise.\n\n"
            "**Sweet spot** — 256–512 tokens (~500–1000 characters) with 10–15% overlap "
            "is a common starting point. The best size depends on your data."
        )

    st.divider()
    st.page_link(
        "pages/04_embeddings.py",
        label="Next: Generate Embeddings",
        icon=":material/arrow_forward:",
    )
