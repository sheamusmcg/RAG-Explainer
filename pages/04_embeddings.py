import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from components.state_manager import has_chunks, has_embeddings, clear_downstream
from components.rag_utils import count_tokens, embed_texts, reduce_dimensions

# ── Prerequisite ─────────────────────────────────────────────────────────
if not has_chunks():
    st.title("Embeddings")
    st.warning("Please chunk your documents first.")
    st.page_link("pages/03_chunking.py", label="Go to Chunking", icon=":material/arrow_back:")
    st.stop()

st.title("Embeddings")
st.write(
    "Now we convert each chunk into a **vector** — a list of numbers that captures its "
    "meaning. Chunks with similar meanings will have similar vectors."
)

chunks = st.session_state["chunks"]
chunk_texts = [c["text"] for c in chunks]

# ── Section 1: Tokenization Preview ──────────────────────────────────────
st.header("Step 1 — Tokenization")
st.write(
    "Before embedding, text is broken into **tokens** (words, sub-words, or characters). "
    "Here's how many tokens each chunk contains:"
)

if st.button("Count Tokens", key="count_tokens"):
    with st.spinner("Counting tokens..."):
        st.session_state["token_counts"] = count_tokens(chunk_texts)

if st.session_state.get("token_counts") is not None:
    token_counts = st.session_state["token_counts"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tokens", sum(token_counts))
    with col2:
        st.metric("Avg Tokens / Chunk", int(np.mean(token_counts)))
    with col3:
        st.metric("Max Tokens in a Chunk", max(token_counts))

    token_df = pd.DataFrame({"Chunk #": range(1, len(token_counts) + 1), "Tokens": token_counts})
    fig = px.bar(
        token_df, x="Chunk #", y="Tokens",
        title="Token Count per Chunk",
        labels={"Tokens": "Token Count"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ── Section 2: Generate Embeddings ───────────────────────────────────────
st.header("Step 2 — Generate Embeddings")

embedding_method = st.radio(
    "Embedding method",
    ["TF-IDF (fast, no downloads)", "Sentence Transformer (better quality, needs torch)"],
    index=0,
    help="TF-IDF is instant and works everywhere. Sentence Transformer produces higher-quality "
         "semantic embeddings but requires ~500MB of downloads (torch + model).",
)
method_key = "tfidf" if "TF-IDF" in embedding_method else "transformer"

if method_key == "tfidf":
    st.info(
        "**TF-IDF** (Term Frequency–Inverse Document Frequency) creates vectors based on word "
        "importance. Words that appear often in a chunk but rarely across all chunks get higher "
        "weight. It's fast and demonstrates the core concept of turning text into numbers."
    )
else:
    st.info(
        "**Sentence Transformers** use a neural network to produce dense vectors that capture "
        "semantic meaning. Two sentences about the same topic — even with different words — "
        "produce similar vectors. Requires downloading ~500MB of model files on first use."
    )

if st.button("Generate Embeddings", type="primary"):
    with st.spinner("Generating embeddings..."):
        embeddings = embed_texts(chunk_texts, method=method_key)
    clear_downstream("embeddings")
    st.session_state["embeddings"] = embeddings
    st.session_state["embedding_method"] = method_key
    st.success(f"Generated **{embeddings.shape[0]}** embeddings of dimension **{embeddings.shape[1]}**.")

# ── Results ──────────────────────────────────────────────────────────────
if has_embeddings():
    embeddings = st.session_state["embeddings"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vectors", embeddings.shape[0])
    with col2:
        st.metric("Dimensions", embeddings.shape[1])
    with col3:
        method_label = "TF-IDF" if st.session_state.get("embedding_method") == "tfidf" else "Transformer"
        st.metric("Method", method_label)

    with st.expander("Preview: First embedding vector (first 20 values)"):
        st.code(str(embeddings[0][:20].tolist()))

    # ── Visualization ────────────────────────────────────────────────────
    st.subheader("Embedding Space Visualization")
    st.write("Each point is a chunk, projected into 2-D. Chunks from the same document share a color.")

    source_labels = [c["source_title"] for c in chunks]

    tab_pca, tab_tsne = st.tabs(["PCA", "t-SNE"])

    with tab_pca:
        coords_pca = reduce_dimensions(embeddings, n_components=2, method="pca")
        df_pca = pd.DataFrame({
            "x": coords_pca[:, 0],
            "y": coords_pca[:, 1],
            "Source": source_labels,
            "Preview": [t[:80] + "..." for t in chunk_texts],
        })
        fig_pca = px.scatter(
            df_pca, x="x", y="y", color="Source", hover_data=["Preview"],
            title="2-D PCA Projection of Chunk Embeddings",
        )
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab_tsne:
        if embeddings.shape[0] < 5:
            st.info("t-SNE needs at least 5 data points. Add more documents or use smaller chunks.")
        else:
            coords_tsne = reduce_dimensions(embeddings, n_components=2, method="tsne")
            df_tsne = pd.DataFrame({
                "x": coords_tsne[:, 0],
                "y": coords_tsne[:, 1],
                "Source": source_labels,
                "Preview": [t[:80] + "..." for t in chunk_texts],
            })
            fig_tsne = px.scatter(
                df_tsne, x="x", y="y", color="Source", hover_data=["Preview"],
                title="2-D t-SNE Projection of Chunk Embeddings",
            )
            fig_tsne.update_layout(height=500)
            st.plotly_chart(fig_tsne, use_container_width=True)

    # ── Learn more ───────────────────────────────────────────────────────
    with st.expander("Learn more: What are Embeddings?"):
        st.write(
            "An **embedding** maps text into a high-dimensional space where distance "
            "corresponds to meaning. Two sentences about the same topic — even using "
            "different words — produce vectors that are close together.\n\n"
            "**TF-IDF** captures word importance but misses synonyms. "
            "**Transformer models** capture deep semantic meaning but are slower.\n\n"
            "Common embedding models:\n"
            "- **TF-IDF** — sparse, keyword-based, fast (used in this demo by default)\n"
            "- **Word2Vec / GloVe** — word-level, older\n"
            "- **BERT / Sentence-BERT** — sentence-level, context-aware\n"
            "- **OpenAI text-embedding-3** — high accuracy, API-based\n"
            "- **all-MiniLM-L6-v2** — fast, lightweight (optional in this demo)"
        )

    st.divider()
    st.page_link(
        "pages/05_indexing.py",
        label="Next: Build Index",
        icon=":material/arrow_forward:",
    )
